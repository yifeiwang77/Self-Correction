from collections import Counter
from typing import Iterable

from analysis.graders.assessment import Assessment
from analysis.metrics.binomial import BinomialDistribution


def calculate_accuracy(
    assessments: Iterable[Assessment],
    confidence_level: float = 0.95,
) -> BinomialDistribution:
    """Compute the fraction of assessments that are correct

    The returned object is a tuple containing the proportion of correct answers and the
    surrounding confidence interval.

    Assessments of UNKOWN are excluded.
    """
    counts = Counter(assessments)
    return BinomialDistribution(
        successes=counts[Assessment.CORRECT],
        samples=counts[Assessment.CORRECT] + counts[Assessment.INCORRECT],
        confidence_level=confidence_level,
    )
