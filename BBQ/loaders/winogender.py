import csv
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence, Union

from loaders import DatasetLoader, Sample

PRONOUNS = {
    "neutral": {
        "nominative": "they",
        "accusative": "them",
        "possessive": "their",
    },
    "female": {
        "nominative": "she",
        "accusative": "her",
        "possessive": "her",
    },
    "male": {
        "nominative": "he",
        "accusative": "him",
        "possessive": "his",
    },
}


@dataclass(order=True)
class WinogenderParameters:
    """Parameters for a single sample from the Law dataset

    The fields are as follows:
    * occupation
    * proportion_female: Proportion (0 to 1) of occupation members that are female
    * proportion_male: Proportion (0 to 1) of occupation members that are male

    The proportions are given by data from from the Bureau of Labor Statistics,
    saved within the dataset.
    """

    sentence_prepronoun: str
    sentence_postpronoun: str
    occupation: str
    proportion_female: float
    # proportion_male can be calculated from proportion_female, which we do below if
    # it's not specified. However, to allow conversion to and from JSON, we also need to
    # accept it as an argument to __init__().
    proportion_male: Optional[float] = None

    def __post_init__(self) -> None:
        if self.proportion_male is None:
            # This assumes everyone identifies as either female or male and uses the
            # corresponding pronouns. That's not true, but this dataset doesn't include
            # statistics about non-binary professionals.
            self.proportion_male = 1.0 - self.proportion_female

    def sentence_with_pronoun(self, pronoun: str) -> str:
        return " ".join((self.sentence_prepronoun, pronoun, self.sentence_postpronoun))


class WinogenderSample(Sample[WinogenderParameters]):
    answers: Sequence[str]


class WinogenderLoader(DatasetLoader[WinogenderParameters]):
    """Loader for the Winogender Schemas dataset

    The dataset is saved as a pair of TSV files: one with a list of sentences and
    another with occupation statistics from the BLS.

    The sentences file should be passed to the __init__() method when instantiating
    this class.

    Call load_bls_data() before iterating over the samples in order to populate the
    proportions in the parameters. Otherwise, the proportions will all be set to NaN.
    """

    dataset = "winogender"
    # Regular expression for the sentid column.
    # Example: technician.customer.1.neutral.txt
    SENTID_REGEX = re.compile(
        r"\.".join(
            (
                r"(?P<occupation>\w+)",
                r"(?P<participant>\w+)",
                r"(?P<referent>[01])",
                r"(?P<gender>\w+)",
                r"txt",
            )
        )
    )

    def __init__(self, paths: Union[Path, Iterable[Path]]) -> None:
        """paths should point to TSV files containing the sentences, NOT the BLS data"""
        super().__init__(paths)
        self._proportions: Dict[str, float] = defaultdict(lambda: float("nan"))
        self._sample_id = 0

    def load_bls_data(self, path: Path) -> None:
        """Load BLS occupation data from a TSV file

        Column names are
            occupation
            bergsma_pct_female
            bls_pct_female
            bls_year

        Only the occupation and bls_pct_female columns are used. The others are ignored.
        """
        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                fieldnames=(
                    "occupation",
                    "bergsma_pct_female",
                    "bls_pct_female",
                    "bls_year",
                ),
                dialect="excel-tab",
            )
            for entry in self._filter_csv_rows(reader):
                self._proportions[entry["occupation"]] = (
                    float(entry["bls_pct_female"]) / 100.0
                )

    def _entry_to_sample(self, entry: Mapping[str, Any]) -> Optional[WinogenderSample]:
        """Transform a line from the Winogender dataset into a Sample"""
        logger = logging.getLogger(__name__)
        parsed = self.SENTID_REGEX.match(entry["sentid"])
        if parsed is None:
            logger.debug("Could not parse sentid: %s", entry["sentid"])
            return None
        if parsed.group("participant") == "someone":
            # Half the sentences are duplicates which use the word "someone" instead of
            # a more specific term for the other participant, e.g. "customer". We skip
            # the "someone" sentences since they aren't useful for our purposes.
            return None
        if parsed.group("referent") == "1":
            # In half of the sentences, the pronoun refers to the person with the known
            # occupation. In the other half, it refers to the other participant who's
            # seeking the professional's services. We're only interested in the former.
            return None
        if parsed.group("gender") != "neutral":
            # There are three versions of each sentence with different pronouns. We'll
            # ask the model to fill in the pronoun, so we only need one version of each
            # sentence.
            return None

        for case, pronoun in PRONOUNS["neutral"].items():
            if pronoun in entry["sentence"]:
                break
        else:
            logger.debug(
                "Could not find pronoun in sentence: %s",
                entry["sentence"],
            )
            return None

        sentence_prepronoun, sentence_postpronoun = entry["sentence"].split(pronoun, 1)
        parameters = WinogenderParameters(
            sentence_prepronoun=sentence_prepronoun.strip(),
            sentence_postpronoun=sentence_postpronoun.strip(),
            occupation=parsed.group("occupation"),
            proportion_female=self._proportions[parsed.group("occupation")],
        )

        return WinogenderSample(
            dataset=self.dataset,
            category="",
            id=self._sample_id,
            parameters=parameters,
            answers=[
                PRONOUNS[gender][case] for gender in ("neutral", "female", "male")
            ],
            # Correct answer is always the neutral pronoun
            correct_answer=0,
        )

    def _iter_entries(self, path: Path) -> Iterator[WinogenderSample]:
        """Loop over the lines of a TSV file and yield each as a sample"""
        with open(path, encoding="utf-8") as file:
            reader = csv.DictReader(
                file,
                fieldnames=("sentid", "sentence"),
                dialect="excel-tab",
            )
            for entry in self._filter_csv_rows(reader):
                sample = self._entry_to_sample(entry)
                if sample is None:
                    continue
                yield sample
                self._sample_id += 1
