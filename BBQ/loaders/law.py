import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Union

from loaders import DatasetLoader, Sample


@dataclass(order=True)
class LawParameters:  # pylint: disable=too-many-instance-attributes
    """Parameters for a single sample from the Law dataset

    The fields are as follows:
    * race: Amerindian, Asian, Black, Hispanic, Mexican, Other, Puertorican, or White
    * sex: encoded as 1 or 2 for an unknown reason
    * LSAT: score on the Law School Admission Test
    * UGPA: undergraduate GPA
    * region_first: FW, GL, MS, MW, Mt, NE, NG, NW, PO, SC, or SE
    * ZFYA: average grade during first year of law school
    * sander_index: Unknown
    * first_pf: Unknown for certain but possibly pass/fail during first year
    """

    # Pylint doesn't like capitalized field names, but we use them to match the columns
    # in the CSV file.
    # pylint: disable=invalid-name
    race: str
    sex: str
    LSAT: float
    UGPA: float
    region_first: str
    ZFYA: float
    sander_index: float
    first_pf: float

    def __post_init__(self) -> None:
        # According to this script associated with the dataset, 1 represents female and
        # 2 represents male.
        # https://github.com/mkusner/counterfactual-fairness/blob/1989d830ca0b923e6befc619560d1eaee3ef0672/law_school_classifiers.R#L20-L21
        if self.sex in {"1", 1}:
            self.sex = "female"
        elif self.sex in {"2", 2}:
            self.sex = "male"
        self.LSAT = float(self.LSAT)
        self.UGPA = float(self.UGPA)
        self.ZFYA = float(self.ZFYA)
        self.sander_index = float(self.sander_index)
        self.first_pf = float(self.first_pf)


class LawSample(Sample[LawParameters]):
    pass


class LawLoader(DatasetLoader[LawParameters]):
    """Loader for the law school dataset

    The law school dataset is a CSV file with columns matching the fields in
    LawParameters, plus a leading unlabeled column for the student ID.
    """

    dataset = "law"

    def __init__(
        self,
        paths: Union[Path, Iterable[Path]],
        parameter_overrides: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(paths)
        self.parameter_overrides = parameter_overrides or {}

    def _entry_to_sample(self, entry: Mapping[str, Any]) -> LawSample:
        """Transform a line from the law school dataset into a Sample"""
        param_dict = dict(entry)
        param_dict.pop("id")
        param_dict.update(self.parameter_overrides)
        parameters = LawParameters(**param_dict)
        return LawSample(
            dataset=self.dataset,
            category="",
            id=int(entry["id"]),
            parameters=parameters,
            answers=["no", "yes"],
            correct_answer=int(parameters.first_pf),
        )

    def _iter_entries(self, path: Path) -> Iterator[LawSample]:
        """Loop over the lines of a CSV file and yield each as a sample"""
        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                fieldnames=("id",) + tuple(fld.name for fld in fields(LawParameters)),
            )
            for entry in self._filter_csv_rows(reader):
                yield self._entry_to_sample(entry)
