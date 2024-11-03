import logging
import re
import string
from typing import Any, Optional

from eval.result import Result

# Examples:
#   "yes. "
#   "(a) correct answer. "
SENTENCE_REGEX = re.compile(r"(?P<sentence>.*\w.*)\.\s")


def result_to_answer(
    result: Result[Any],
    strip_punctuation: bool = True,
) -> Optional[str]:
    """Extract the model's answer from a result and clean it up a bit

    The answer is converted to lowercase, and all leading and trailing whitespace is
    removed. If strip_punctuation is True, leading and trailing punctuation is also
    removed.
    """
    logger = logging.getLogger(__name__)
    try:
        answer = result.reply.choices[0].message.content
    except IndexError as err:
        logger.debug("Could not find answer in result due to %r", err)
        return None

    # If the model replied with multiple sentences, only return the first
    sentence_match = SENTENCE_REGEX.search(answer)
    if sentence_match:
        answer = sentence_match["sentence"]

    chars_to_strip = string.whitespace
    if strip_punctuation:
        chars_to_strip += string.punctuation
    model_answer = answer.strip(chars_to_strip).lower()
    return model_answer
