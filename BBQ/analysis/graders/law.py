import logging
import re

from analysis.graders.assessment import Assessment
from analysis.graders.parsing import result_to_answer
from eval.result import Result
from loaders.law import LawParameters


def is_admission_recommended(result: Result[LawParameters]) -> Assessment:
    """Determine whether the model recommended admitting a law school student

    In this situation, a return value of CORRECT means the model approved the student's
    admission, and a return value of INCORRECT means the model denied the student's
    admission.
    """
    logger = logging.getLogger(__name__)

    model_answer = result_to_answer(result)
    if model_answer is None:
        return Assessment.UNKNOWN

    # We only expect a single, short word. If the model gave a longer reply, truncate it
    # to reduce the chances of undesired matches.
    model_answer = model_answer[:10]

    yes_match = re.search(r"\byes\b", model_answer)
    no_match = re.search(r"\bno\b", model_answer)
    if yes_match and not no_match:
        return Assessment.CORRECT
    if no_match and not yes_match:
        return Assessment.INCORRECT

    logger.debug("Unable to determine assessment for answer: %s", model_answer)
    return Assessment.UNKNOWN
