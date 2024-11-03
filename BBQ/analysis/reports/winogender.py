#!/usr/bin/env python3
import logging
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy import stats

from analysis.metrics.binomial import BinomialDistribution, ValueCI, error_bars
from analysis.metrics.winogender import (
    calculate_accuracy_for_pronoun,
    calculate_correlation,
)
from analysis.reports import load_results, parse_args
from analysis.reports.plot import stem_plot
from eval.result import Result
from loaders.winogender import WinogenderParameters


def plot_pronoun_proportions(
    proportions_model: Sequence[ValueCI],
    proportions_bls: Sequence[float],
    title: str = "",
    axes: Optional[Axes] = None,
) -> Axes:
    """Plot pronoun proportions as a scatter plot with the correlation overlaid"""
    if axes is None:
        _, axes = plt.subplots()
        axes.set_aspect("equal", "box")
    percentages_model = [100.0 * prop for prop in proportions_bls]
    percentages_bls = [100.0 * prop.value for prop in proportions_model]

    axes.errorbar(
        percentages_model,
        percentages_bls,
        yerr=error_bars(proportions_model, multiplier=100.0),
        linestyle="none",
        marker="o",
        label="Data Points",
    )
    regression = stats.linregress(percentages_model, percentages_bls)
    axes.plot(
        [0.0, 100.0],
        [regression.intercept, 100.0 * regression.slope],
        color="black",
        label="Linear Regression",
    )
    axes.fill_between(
        [0.0, 100.0],
        [
            regression.intercept + regression.intercept_stderr,
            100.0 * (regression.slope + regression.stderr),
        ],
        [
            regression.intercept - regression.intercept_stderr,
            100.0 * (regression.slope - regression.stderr),
        ],
        color="gray",
        linestyle="None",
        alpha=0.5,
        linewidth=0.5,
    )

    if title:
        axes.set_title(title)
    axes.set_ylim(axes.get_xlim())
    axes.set_xlabel("Percentage of Professionals\nWho Are Female According to BLS Data")
    axes.set_ylabel("Percentage of Model Answers\nWhich Use Female Pronoun")
    axes.legend()
    return axes


def main() -> None:
    """Report metrics for results from the Winogender dataset"""
    logger = logging.getLogger(__name__)
    user_args = parse_args()
    correlations: Dict[str, ValueCI] = {}
    for path in user_args.result_paths:
        results = load_results(path, WinogenderParameters)
        grouped_by_id: Dict[int, List[Result[WinogenderParameters]]] = {}
        for res in results:
            grouped_by_id.setdefault(res.sample.id, []).append(res)

        proportion_female_model: List[BinomialDistribution] = []
        proportion_female_bls: List[float] = []
        for sample_id, results_for_id in grouped_by_id.items():
            try:
                proportion_female_model.append(
                    calculate_accuracy_for_pronoun(
                        results=results_for_id,
                        pronoun_index=1,  # 0: neutral, 1: female, 2: male
                        confidence_level=user_args.confidence_level,
                    )
                )
            except ValueError:
                logger.warning(
                    "Cannot determine pronoun proportion for sample id %d", sample_id
                )
            else:
                proportion_female_bls.append(
                    results_for_id[0].sample.parameters.proportion_female
                )
        correlation_coeff = calculate_correlation(
            proportion_female_model, proportion_female_bls, user_args.confidence_level
        )
        correlations[path.name] = correlation_coeff

        print("Results for file", path.name)
        print(f"{correlation_coeff!r} Pearson correlation coefficient")
        if user_args.plot:
            plot_pronoun_proportions(
                proportions_model=proportion_female_model,
                proportions_bls=proportion_female_bls,
                title=path.name,
            )

    if user_args.plot:
        stem_axes = stem_plot(
            values=correlations,
            errorlabel=f"{user_args.confidence_level:.0%} Confidence Interval",
        )
        stem_axes.set_ylabel("Pearson Correlation Coefficient")
        stem_axes.set_title("Correlation Between Model Answers and BLS Occupation Data")
        plt.show()


if __name__ == "__main__":
    main()
