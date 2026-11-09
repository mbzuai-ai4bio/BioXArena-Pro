from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_layout import (  # noqa: E402
    DOMAIN_TASKS,
    LEGACY_TO_CANONICAL_TASK_KEY,
    PARTITION_DOMAINS,
    TASK_KEY_SET,
    canonicalize_task_key,
    partition_task_keys,
)


class BenchmarkLayoutTests(unittest.TestCase):
    def test_exactly_eight_domains_with_eight_tasks_each(self) -> None:
        self.assertEqual(len(DOMAIN_TASKS), 8)
        self.assertEqual({len(tasks) for tasks in DOMAIN_TASKS.values()}, {8})
        self.assertEqual(len(TASK_KEY_SET), 64)

    def test_four_partitions_are_disjoint_and_complete(self) -> None:
        seen: set[str] = set()
        for part in sorted(PARTITION_DOMAINS):
            keys = partition_task_keys(part)
            self.assertEqual(len(keys), 16)
            self.assertEqual(len(set(keys)), 16)
            self.assertFalse(seen.intersection(keys))
            seen.update(keys)
        self.assertEqual(seen, TASK_KEY_SET)

    def test_code_task_tree_matches_canonical_layout(self) -> None:
        code_tasks = {
            f"{path.parents[1].name}/{path.parent.name}"
            for path in (PROJECT_ROOT / "tasks").glob("*/*/grade.py")
        }
        self.assertEqual(code_tasks, TASK_KEY_SET)

    def test_legacy_task_selectors_map_to_canonical_keys(self) -> None:
        for legacy, canonical in LEGACY_TO_CANONICAL_TASK_KEY.items():
            self.assertEqual(canonicalize_task_key(legacy), canonical)
            self.assertIn(canonical, TASK_KEY_SET)


if __name__ == "__main__":
    unittest.main()
