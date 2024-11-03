import math
from typing import List, Optional, Sequence, Tuple

from scipy import stats


class ValueCI:
    """A value and associated confidence interval"""

    def __init__(
        self,
        value: float,
        confidence_level: float,
        confidence_interval: Tuple[float, float],
    ) -> None:
        self._value = value
        self.confidence_level = confidence_level
        self.confidence_interval = confidence_interval

    @property
    def value(self) -> float:
        return self._value

    @property
    def ci_low(self) -> float:
        return float(self.confidence_interval[0])

    @property
    def ci_high(self) -> float:
        return float(self.confidence_interval[1])

    @property
    def ci_low_rel(self) -> float:
        return self.ci_low - self.value

    @property
    def ci_high_rel(self) -> float:
        return self.ci_high - self.value

    def __str__(self) -> str:
        return f"{self.value:.3f}"

    def __repr__(self) -> str:
        return " ".join(
            (
                # Use a width of 6 so things line up nicely with BinomialDistribution
                # strings too.
                f"{self.value:6.3f}",
                f"({self.confidence_level:3.0%} CI:",
                f"{self.ci_low:.3f} - {self.ci_high:.3f})",
            )
        )


class BinomialDistribution(ValueCI):
    """A binomial distribution with an associated confidence interval

    The interval is an absolute one:
        interval_low <= value <= interval_high

    An relative interval can be obtained with ci_low_rel and ci_high_rel.
    """

    # pylint: disable-next=super-init-not-called
    def __init__(
        self,
        successes: int,
        samples: int,
        confidence_level: float,
        confidence_interval: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.binom_test_result = stats.binomtest(successes, samples)
        self.confidence_level = confidence_level
        if confidence_interval is not None:
            self.confidence_interval = confidence_interval
        else:
            self.confidence_interval = self.binom_test_result.proportion_ci(
                confidence_level=self.confidence_level,
                method="exact",
            )

    @property
    def value(self) -> float:
        return float(self.binom_test_result.statistic)

    @property
    def proportion(self) -> float:
        """An alias for self.value

        The term proportion is more specific and intuitive in the case of binomial
        distributions, where we're dealing with a ratio of successes and samples.
        """
        return self.value

    def __str__(self) -> str:
        # Since self.value lies in the range [0, 1], format it as a percentage
        return f"{self.value:6.1%}"

    def __repr__(self) -> str:
        return " ".join(
            (
                f"{self.value:6.1%}",
                f"({self.confidence_level:3.0%} CI:",
                f"{self.ci_low:6.1%} - {self.ci_high:6.1%})",
            )
        )


def binomial_difference(
    binom1: BinomialDistribution,
    binom2: BinomialDistribution,
) -> BinomialDistribution:
    """Compute the difference in proportions of two binomial distributions

    Warning: do not pass the result of this function back to it a second time. Doing so
    will produce inaccurate values.

    This uses Newcombe's interval discussed in Brown and Li:
    http://stat.wharton.upenn.edu/~lbrown/Papers/2005c%20Confidence%20intervals%20for%20the%20two%20sample%20binomial%20distribution%20problem.pdf

    The interval was originally published by Newcombe in:
    https://www.researchgate.net/publication/13687790_Interval_estimation_for_the_difference_between_independent_proportions_Comparison_of_eleven_methods
    """
    if binom1.confidence_level != binom2.confidence_level:
        raise ValueError("The confidence levels of the two distributions must be equal")
    # Newcombe's interval relies on the Wilson intervals of the two inputs, which isn't
    # what the BinomialDistribution uses by default. Here we compute them ourselves.
    low1, high1 = binom1.binom_test_result.proportion_ci(
        confidence_level=binom1.confidence_level,
        method="wilson",
    )
    low2, high2 = binom2.binom_test_result.proportion_ci(
        confidence_level=binom2.confidence_level,
        method="wilson",
    )

    successes1 = binom1.binom_test_result.k
    samples1 = binom1.binom_test_result.n
    successes2 = binom2.binom_test_result.k
    samples2 = binom2.binom_test_result.n

    z_value = stats.norm.ppf(1 - (1 - binom1.confidence_level) / 2)
    low_diff = z_value * math.sqrt(
        low1 * (1 - low1) / samples1 + high2 * (1 - high2) / samples2
    )
    high_diff = z_value * math.sqrt(
        high1 * (1 - high1) / samples1 + low2 * (1 - low2) / samples2
    )

    # The number of successes and samples don't have much meaning for the difference
    # between two proportions. We take an approach that produces the correct proportion
    # for the difference, but wouldn't be appropriate for feeding into this function
    # again.
    return BinomialDistribution(
        successes=successes1 * samples2 - successes2 * samples1,
        samples=samples1 * samples2,
        confidence_level=binom1.confidence_level,
        confidence_interval=(
            max(min(binom1.proportion - binom2.proportion - low_diff, 1.0), 0.0),
            max(min(binom1.proportion - binom2.proportion + high_diff, 1.0), 0.0),
        ),
    )


def error_bars(
    ratios: Sequence[ValueCI],
    multiplier: float = 100.0,
) -> Tuple[List[float], List[float]]:
    """Return error bars compatible with Matplotlib

    The error bars are multiplied by the given multiplier, which defaults to 100.0 to
    convert proportions to percentages.
    """
    return (
        # Matplotlib expects both sides of the error bar to be positive, which is why we
        # take the absolute value of the relative confidence interval.
        [abs(r.ci_low_rel * multiplier) for r in ratios],
        [abs(r.ci_high_rel * multiplier) for r in ratios],
    )
