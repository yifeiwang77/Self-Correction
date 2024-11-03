import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Type

import jsonlines

from eval.result import Completion, Reply, Result
from loaders import P, Sample
from prompts.message import Message


@dataclass
class UserArguments:
    result_paths: Sequence[Path]
    confidence_level: float
    plot: bool


def parse_args() -> UserArguments:
    """Parse command line arguments

    If no arguments are given (or the --help option is present), this will print out
    usage information and then exit.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_paths",
        type=Path,
        nargs="+",
        help="Paths to JSONL files containing results",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level (0 to 1) for intervals (default: 0.95)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot analysis metrics",
    )
    args = parser.parse_args()
    return UserArguments(**vars(args))


def load_results(
    path: Path,
    # This function uses PascalCase for one of its parameters since it refers to a
    # class, rather than a particular instance.
    # pylint: disable-next=invalid-name
    ParameterClass: Type[P],
    split_completions: bool = True,
) -> Iterable[Result[P]]:
    """Load results from a JSONL file

    ParameterClass should be the dataclass that corresponds to the samples of this
    dataset, e.g. BBQParameters if these results were generated with the BBQ dataset.

    If split_completions is True, any Result objects which contain multiple completions
    are turned into separate Results, each with one completion. In otherwords, rather
    than yielding one Result with N completions, yield N Results each with one
    completion.
    """
    logger = logging.getLogger(__name__)
    with jsonlines.open(path) as reader:
        for entry in reader:
            try:
                entry["sample"]["parameters"] = ParameterClass(
                    **entry["sample"]["parameters"]
                )
                sample = Sample[P](**entry["sample"])
                prompt_messages = [
                    Message(**message) for message in entry["prompt_messages"]
                ]
                completions: List[Completion] = []
                for compl in entry["reply"]["choices"]:
                    compl["message"] = Message(**compl["message"])
                    completions.append(Completion(**compl))
                completions_per_request: Iterable[List[Completion]] = [completions]
                if split_completions:
                    completions_per_request = ([compl] for compl in completions)

                for compls in completions_per_request:
                    entry["reply"]["choices"] = compls
                    reply = Reply(**entry["reply"])
                    yield Result(
                        sample=sample, prompt_messages=prompt_messages, reply=reply
                    )
            except KeyError as err:
                logger.debug("Skipping entry due to error: %r", err)
