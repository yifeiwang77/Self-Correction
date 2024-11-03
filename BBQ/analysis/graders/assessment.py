from enum import Enum, unique
from typing import Optional


@unique
class Assessment(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNKNOWN = "unknown"


def grade_answer(model_answer: Optional[int], correct_answer: int) -> Assessment:
    """Determine whether the model's answer matches the correct one"""
    if model_answer is None:
        return Assessment.UNKNOWN
    if model_answer == correct_answer:
        return Assessment.CORRECT
    return Assessment.INCORRECT
