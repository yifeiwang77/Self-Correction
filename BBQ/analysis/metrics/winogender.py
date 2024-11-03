from typing import Iterable, Sequence

from scipy import stats

from analysis.graders.winogender import is_answer_correct
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.binomial import BinomialDistribution, ValueCI
from eval.result import Result
from loaders.winogender import WinogenderParameters


def calculate_accuracy_for_pronoun(
    results: Iterable[Result[WinogenderParameters]],
    pronoun_index: int,
    confidence_level: float = 0.95,
) -> BinomialDistribution:
    """Determine how often the model chose a given pronoun"""
    return calculate_accuracy(
        assessments=(
            is_answer_correct(res, correct_answer=pronoun_index) for res in results
        ),
        confidence_level=confidence_level,
    )


def calculate_correlation(
    proportions_model: Sequence[ValueCI],
    proportions_bls: Sequence[float],
    confidence_level: float = 0.95,
) -> ValueCI:
    """Calculate the correlation between the model's answers and BLS data

    Compute the correlation using Pearson's coefficient.
    """
    pearson = stats.pearsonr(
        [prop.value for prop in proportions_model],
        proportions_bls,
    )
    return ValueCI(
        value=pearson.statistic,
        confidence_level=confidence_level,
        confidence_interval=pearson.confidence_interval(confidence_level),
    )
