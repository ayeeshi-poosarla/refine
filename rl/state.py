"""RubricState — in-memory representation of a rubric + filled records.

Enables GRPO-compatible rollouts by supporting copy-on-apply branching.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

SPLITS = ("train", "val", "test")


@dataclass
class RubricState:
    task: str
    rubric: dict                          # full rubric.json contents
    records: dict[str, list[dict]]        # {"train": [...], "val": [...], "test": [...]}

    @classmethod
    def from_disk(cls, task: str, rubric_dir: Path,
                  rubricified_dir: Path) -> "RubricState":
        rubric_path = Path(rubric_dir) / task / "rubric.json"
        rubric = json.load(open(rubric_path))

        records: dict[str, list[dict]] = {}
        for split in SPLITS:
            path = Path(rubricified_dir) / task / f"{split}.json"
            if path.exists():
                records[split] = json.load(open(path))

        return cls(task=task, rubric=rubric, records=records)

    def to_disk(self, rubric_dir: Path, rubricified_dir: Path) -> None:
        rubric_dir = Path(rubric_dir)
        rubricified_dir = Path(rubricified_dir)

        out = rubric_dir / self.task
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "rubric.json", "w") as f:
            json.dump(self.rubric, f, indent=2)

        for split, recs in self.records.items():
            out = rubricified_dir / self.task
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"{split}.json", "w") as f:
                json.dump(recs, f, indent=2)

    def copy(self) -> "RubricState":
        return RubricState(
            task=self.task,
            rubric=copy.deepcopy(self.rubric),
            records=copy.deepcopy(self.records),
        )
