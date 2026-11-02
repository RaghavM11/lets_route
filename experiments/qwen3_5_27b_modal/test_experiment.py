from __future__ import annotations

import unittest
from pathlib import Path

from experiment import (
    DEFAULT_TASKS,
    extract_mc_choice,
    load_queries,
    safe_run_id,
    select_queries,
    xroute_messages,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class ExperimentTest(unittest.TestCase):
    def test_choice_extraction_is_strict(self) -> None:
        self.assertEqual(extract_mc_choice("C"), "C")
        self.assertEqual(extract_mc_choice("Final answer: (B)"), "B")
        self.assertEqual(extract_mc_choice("The final answer is D."), "D")
        self.assertEqual(extract_mc_choice("Answer: **E**"), "E")
        self.assertIsNone(extract_mc_choice("A is tempting, but C is better"))
        self.assertIsNone(
            extract_mc_choice("Options include (A) first and (B) second, because")
        )

    def test_prompt_role_split(self) -> None:
        messages = xroute_messages(
            "[System Instruction]be precise[User Query]pick one"
        )
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be precise"},
                {"role": "user", "content": "pick one"},
            ],
        )

    def test_real_data_selection_is_deterministic(self) -> None:
        table, _, audit = load_queries(REPO_ROOT / "data/raw", "test")
        first = select_queries(table, DEFAULT_TASKS, per_task=2, seed=42)
        second = select_queries(table, DEFAULT_TASKS, per_task=2, seed=42)
        self.assertEqual(audit["candidate_models"], 18)
        self.assertEqual(len(first), 12)
        self.assertListEqual(
            first["embedding_id"].tolist(), second["embedding_id"].tolist()
        )

    def test_run_id_validation(self) -> None:
        self.assertEqual(safe_run_id("trial-01"), "trial-01")
        with self.assertRaises(ValueError):
            safe_run_id("../escape")


if __name__ == "__main__":
    unittest.main()
