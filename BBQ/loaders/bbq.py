import csv
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

import jsonlines

from loaders import DatasetLoader, Sample

# Sample IDs are not unique across the dataset. Questions are uniquely identified by the
# combination of category, question_index and example_id.
QuestionKey = Tuple[str, int]


@unique
class BBQPolarity(Enum):
    NEGATIVE = "neg"
    NONNEGATIVE = "nonneg"


@unique
class BBQContextCondition(Enum):
    AMBIGUOUS = "ambig"
    DISAMBIGUOUS = "disambig"


@dataclass(order=True)
class BBQParameters:
    """Parameters for a single Sample from the BBQ dataset

    The fields are as follows:
    * context: A sentence describing the context of the question
    * context_condition: "ambig" or "disambig", depending on the detail of the context
    * polarity: "pos" or "neg", depending on the stereotype the context describes
    """

    context: str
    context_condition: BBQContextCondition
    polarity: BBQPolarity
    question: str
    bias_target_index: Optional[int]

    def __post_init__(self) -> None:
        self.context_condition = BBQContextCondition(self.context_condition)
        self.polarity = BBQPolarity(self.polarity)


class BBQSample(Sample[BBQParameters]):
    pass


class BBQLoader(DatasetLoader[BBQParameters]):
    """Loader for the Bias Benchmark for QA (BBQ) dataset

    The BBQ dataset is saved as a series of JSONL files, one for each category.

    Call load_bias_targets() before iterating over the samples in order to populate the
    index specifying which answer targets the particular social bias the question is
    probing. Otherwise, the target indices will all be set to None.
    """

    dataset = "bbq"

    def __init__(self, paths: Union[Path, Iterable[Path]]) -> None:
        """paths should point to JSONL files containing the questions

        Social bias target data must be loaded separately via load_bias_targets().
        """
        super().__init__(paths)
        self._bias_targets: Dict[QuestionKey, Optional[int]] = {}

    def load_bias_targets(self, path: Path) -> None:
        """Load social bias target information from a CSV file

        Each row must start with the following columns. Additional columns may be
        present, but they will be ignored.
            category
            question_index
            example_id
            target_loc
        """
        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                fieldnames=(
                    "category",
                    "question_index",
                    "example_id",
                    "target_loc",
                ),
            )
            for i, entry in enumerate(reader):
                if i == 0 and all(k == v for k, v in entry.items() if k is not None):
                    # Skip header row
                    continue

                key = (entry["category"].lower(), int(entry["example_id"]))

                try:
                    target_index = int(entry["target_loc"])
                except ValueError:
                    # Some entries have target_loc set to "NA"
                    target_index = None

                # Some keys are duplicated. Check that they have the same target index.
                if key in self._bias_targets:
                    if not self._bias_targets[key] == target_index:
                        raise ValueError(
                            f"Duplicate key {key} with different target indices: "
                            f"{self._bias_targets[key]} != {target_index}"
                        )
                else:
                    self._bias_targets[key] = target_index

    def _entry_to_sample(self, entry: Mapping[str, Any]) -> BBQSample:
        """Transform a line from the BBQ dataset into a Sample"""
        parameters = BBQParameters(
            context=entry["context"],
            context_condition=entry["context_condition"],
            polarity=entry["question_polarity"],
            question=entry["question"],
            bias_target_index=self._bias_targets.get(
                (entry["category"].lower(), int(entry["example_id"]))
            ),
        )
        return BBQSample(
            dataset=self.dataset,
            category=entry["category"].lower(),
            id=entry["example_id"],
            parameters=parameters,
            answers=[
                entry["ans0"],
                entry["ans1"],
                entry["ans2"],
            ],
            correct_answer=entry["label"],
        )

    def _iter_entries(self, path: Path) -> Iterator[BBQSample]:
        """Loop over the lines of a JSONL file and yield each as a sample"""
        with jsonlines.open(path) as reader:
            for entry in reader:
                yield self._entry_to_sample(entry)
