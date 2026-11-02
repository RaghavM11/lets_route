import unittest
import json
from pathlib import Path
import tempfile

from smoke_worker import messages, score_response
from experiments.sglang_modal.fixed_tasks_app import resume_runs


class SmokeScoringTest(unittest.TestCase):
    def test_mmlu_pro_answer_i_and_format(self):
        task = {"metric": "em_mc", "ground_truth": "I"}
        self.assertEqual(score_response(task, "I. Ransom Trojan")["score"], 1)
        self.assertEqual(score_response(task, "Final answer: (J)")["score"], 0)

    def test_final_mc_choice_overrides_discussion(self):
        task = {"metric": "em_mc", "ground_truth": "B"}
        self.assertEqual(score_response(task, "(A) is tempting. Final answer: (B).")["score"], 1)
        self.assertEqual(score_response(task, "I cannot tell.")["score"], 0)

    def test_numeric_reference_and_boxed_result(self):
        self.assertEqual(score_response({"metric": "GSM8K", "ground_truth": "8+10+9\n#### 27"}, "Total: 27")["score"], 1)
        task = {"metric": "MATH", "ground_truth": "Maximum is \\boxed{729}."}
        self.assertEqual(score_response(task, "Therefore \\boxed{729}.")["score"], 1)
        self.assertEqual(score_response(task, "1260")["score"], 0)

    def test_f1_preserves_partial_credit(self):
        task = {"metric": "f1_score", "ground_truth": "ringers"}
        self.assertEqual(score_response(task, "The ringers.")["score"], 1)
        self.assertAlmostEqual(score_response(task, "planted ringers")["score"], 2/3)

    def test_prompt_roles(self):
        self.assertEqual(messages("[System Instruction]be brief[User Query]hello"),
                         [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}])

    def test_resume_preserves_completed_outputs_and_rejects_changed_inputs(self):
        metadata = {"manifest_sha256": "original", "max_new_tokens": 1024,
                    "temperature": 0, "seed": 42, "thinking_enabled": False,
                    "tiers": ["0.8b", "7b", "27b", "72b"]}
        run = {"server_error": None, "rows": [{"fixed_id": str(i)} for i in range(10)]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/"runs.json").write_text(json.dumps({"metadata": metadata, "runs": [run]}))
            runs, resumed = resume_runs(folder, metadata, ["72b"])
            self.assertEqual(runs, [run])
            self.assertEqual(resumed["expected_results"], 40)
            with self.assertRaisesRegex(ValueError, "manifest_sha256"):
                resume_runs(folder, {**metadata, "manifest_sha256": "changed"}, ["72b"])

    def test_resume_rejects_incomplete_tier(self):
        metadata = {"manifest_sha256": "original", "max_new_tokens": 1024,
                    "temperature": 0, "seed": 42, "thinking_enabled": False, "tiers": ["0.8b"]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/"runs.json").write_text(json.dumps({"metadata": metadata,
                "runs": [{"server_error": "crashed", "rows": []}]}))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resume_runs(folder, metadata, ["0.8b"])


if __name__ == "__main__":
    unittest.main()
