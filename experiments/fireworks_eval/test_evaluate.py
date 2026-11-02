import unittest
from unittest.mock import Mock, patch
import hashlib
import json
from pathlib import Path
import tempfile

from experiments.fireworks_eval.client import FireworksClient, FireworksError, CHAT_ENDPOINT
from experiments.fireworks_eval.evaluate import estimate_cost, invoke, record_result
from experiments.fireworks_eval import evaluate


class FireworksEvaluationTest(unittest.TestCase):
    def test_key_redaction_and_endpoint_allowlist(self):
        client=FireworksClient(key='test-secret-key')
        self.assertEqual(client.sanitize({'error':'test-secret-key'}),{'error':'[REDACTED]'})
        client._session.request=Mock()
        with self.assertRaises(ValueError):
            client.request('POST','https://example.com',payload={})
        client._session.request.assert_not_called()
        self.assertFalse(client._session.trust_env)
        self.assertNotIn('x-openai-api-key',client._session.headers)
        self.assertNotIn('x-anthropic-api-key',client._session.headers)

    def test_http_error_does_not_expose_key(self):
        client=FireworksClient(key='test-secret-key')
        response=Mock(status_code=401)
        response.json.return_value={'error':'invalid test-secret-key'}
        client._session.request=Mock(return_value=response)
        with self.assertRaises(FireworksError) as caught:
            client.request('POST',CHAT_ENDPOINT,payload={})
        self.assertNotIn('test-secret-key',str(caught.exception))
        self.assertFalse(client._session.request.call_args.kwargs['allow_redirects'])

    def test_auto_records_actual_serving_model(self):
        task={'fixed_id':'test','query_uid':'hash','task_name':'mmlu','metric':'em_mc','query':'question','ground_truth':'A'}
        response={'model':'glm-5p3-flash','choices':[{'message':{'content':'(A)'},'finish_reason':'stop'}],
                  'usage':{'prompt_tokens':100,'completion_tokens':5}}
        record=record_result(task,'auto',response,[{'status':200,'latency_s':1.2}],None,{'reasoning_effort':'low'})
        self.assertEqual(record['requested_model'],'auto')
        self.assertEqual(record['served_model'],'glm-5p3-flash')
        self.assertTrue(record['auto_model_resolved'])
        self.assertEqual(record['score'],1)

    def test_cost_uses_cache_discount_and_unknown_rates_remain_unknown(self):
        usage={'prompt_tokens':1000,'completion_tokens':100,'prompt_tokens_details':{'cached_tokens':500}}
        self.assertAlmostEqual(estimate_cost(usage,'accounts/fireworks/models/qwen3p8-max'),(500*2+500*.25+100*6)/1e6)
        self.assertAlmostEqual(estimate_cost(usage,'Qwen 3.8 Max'),(500*2+500*.25+100*6)/1e6)
        self.assertIsNone(estimate_cost(usage,'unknown-new-model'))

    def test_reasoning_fallback_is_explicit(self):
        client=Mock()
        client.chat.side_effect=[FireworksError(400,'Unsupported reasoning effort none'),{'model':'test','choices':[]}]
        response,attempts,error,effective=invoke(client,{'model':'test','reasoning_effort':'none'})
        self.assertIsNone(error)
        self.assertEqual(effective['reasoning_effort'],'low')
        self.assertEqual([a['status'] for a in attempts],[400,200])

    def test_authentication_failure_is_not_retried(self):
        client=Mock()
        client.chat.side_effect=FireworksError(401,'invalid key')
        with self.assertRaises(FireworksError): invoke(client,{'model':'auto'})
        self.assertEqual(client.chat.call_count,1)

    def test_retry_errors_keeps_successes_and_archives_failure(self):
        source=(evaluate.ROOT/'fixtures/smoke10/tasks.json').read_bytes()
        tasks=json.loads(source)['tasks']
        response={'model':'glm-5p3-flash','choices':[{'message':{'content':'(A)'},'finish_reason':'stop'}],
                  'usage':{'prompt_tokens':100,'completion_tokens':3}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest=root/'fixtures/smoke10/tasks.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_bytes(source)
            folder=root/'results/fireworks_fixed_ten/retry-test'
            folder.mkdir(parents=True)
            rows=[record_result(task,'auto',response,[{'status':200,'latency_s':1}],None,{'reasoning_effort':'low'}) for task in tasks]
            rows[3]=record_result(tasks[3],'auto',None,[{'status':503,'latency_s':1}],"HTTP 503",{'reasoning_effort':'low'})
            metadata={'arms':['auto'],'max_tokens':1024,'manifest_sha256':hashlib.sha256(source).hexdigest(),
                      'temperature':0,'seed':42,'created_at':'test'}
            (folder/'results.json').write_text(json.dumps(rows))
            (folder/'summary.json').write_text(json.dumps(metadata))
            client=Mock()
            client.chat.return_value=response
            client.sanitize.side_effect=lambda value:value
            with patch.object(evaluate,'ROOT',root),patch.object(evaluate,'FireworksClient',return_value=client),\
                 patch('sys.argv',['evaluate','--arms','auto','--run-id','retry-test','--retry-errors','--retry-reasoning-effort','none']):
                evaluate.main()
            self.assertEqual(client.chat.call_count,1)
            self.assertEqual(len(json.loads((folder/'failed_request_history.json').read_text())),1)
            saved=json.loads((folder/'results.json').read_text())
            self.assertEqual(len(saved),10)
            self.assertTrue(all(row['error'] is None for row in saved))
            retried=next(row for row in saved if row['fixed_id']==tasks[3]['fixed_id'])
            self.assertEqual(retried['effective_request']['reasoning_effort'],'none')
            self.assertTrue((folder/'retry_settings_history.json').exists())


if __name__=='__main__': unittest.main()
