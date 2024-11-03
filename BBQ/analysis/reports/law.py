#!/usr/bin/env python3
from collections import defaultdict
from itertools import chain, combinations
from math import comb
from typing import Dict, List, Optional

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from analysis.graders.assessment import Assessment
from analysis.graders.law import is_admission_recommended
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.binomial import BinomialDistribution, binomial_difference
from analysis.reports import UserArguments, load_results, parse_args
from analysis.reports.plot import add_axhlines, stem_plot
from loaders.law import LawParameters


def plot_admission_rates(
    admission_rates: Dict[str, Dict[str, BinomialDistribution]],
    user_args: UserArguments,
    axes: Optional[Axes] = None,
) -> Axes:
    """Plot admission rates by results file and race using a bar chart"""
    if axes is None:
        _, axes = plt.subplots()
    spacing = 0.1

    for i, (race, rates_by_path) in enumerate(admission_rates.items()):
        xoffset = i * spacing - len(admission_rates) * spacing / 2.0
        xcoords = [j + xoffset for j in range(len(user_args.result_paths))]
        stem_plot(
            values=rates_by_path,
            axhlines=[],
            xcoords=xcoords,
            axes=axes,
            label=f"Race: {race}",
            linefmt=f"C{i}-",
            markerfmt=f"C{i}o",
        )
    add_axhlines(axes, [0.0, 1.0])
    axes.set_ylabel("Admission Rate")
    axes.set_title("Admission Rates for Law School Dataset")
    return axes


def plot_bias(
    admission_rates: Dict[str, Dict[str, BinomialDistribution]],
    user_args: UserArguments,
    axes: Optional[Axes] = None,
) -> Axes:
    """Plot the discrimination bias by results file using a bar chart

    The discrimination bias is the difference in admission rates between two races.
    """
    if axes is None:
        _, axes = plt.subplots()
    num_combinations = comb(len(admission_rates), 2)
    spacing = 0.1

    race_pairs = combinations(admission_rates.keys(), 2)
    for i, (race_a, race_b) in enumerate(race_pairs):
        xoffset = spacing * (i - num_combinations / 2.0)
        xcoords = [j + xoffset for j in range(len(user_args.result_paths))]

        rates_a = [admission_rates[race_a][p.name] for p in user_args.result_paths]
        rates_b = [admission_rates[race_b][p.name] for p in user_args.result_paths]
        biases = [binomial_difference(a, b) for a, b in zip(rates_a, rates_b)]

        stem_plot(
            values=dict(zip(map(str, user_args.result_paths), biases)),
            axhlines=[],
            xcoords=xcoords,
            axes=axes,
            label=f"Bias: {race_a} - {race_b}",
        )
    add_axhlines(axes, [-1.0, 0.0, 1.0])
    axes.set_ylabel("Admission Rate Bias")
    axes.set_title("Discrimination Bias for Law School Dataset")
    return axes


def main() -> None:
    """Report metrics for results from the law school dataset"""
    user_args = parse_args()
    admission_rates: Dict[str, Dict[str, BinomialDistribution]] = {}
    for path in user_args.result_paths:
        assessments_by_race: Dict[str, List[Assessment]] = {}
        for result in load_results(path, LawParameters):
            assessment = is_admission_recommended(result)
            race = result.sample.parameters.race
            if race not in assessments_by_race:
                assessments_by_race[race] = []
            assessments_by_race[race].append(assessment)

        print("Results for file", path.name)
        overall_admission_rate = calculate_accuracy(
            chain.from_iterable(assessments_by_race.values()),
            confidence_level=user_args.confidence_level,
        )
        print(f"{overall_admission_rate!r} admission rate overall")
        for race, assessments in assessments_by_race.items():
            rate = calculate_accuracy(
                assessments,
                confidence_level=user_args.confidence_level,
            )
            print(f"{rate!r} admission rate with race={race}")
            if race not in admission_rates:
                admission_rates[race] = defaultdict(
                    lambda: BinomialDistribution(
                        successes=0, samples=1, confidence_level=1.0
                    )
                )
            admission_rates[race][path.name] = rate

    if user_args.plot:
        _, subplot_axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        rate_axes = plot_admission_rates(
            admission_rates, user_args, axes=subplot_axes[0]
        )
        # For this pair of subplots, we only want the bottom one to have an X label
        rate_axes.set_xlabel("")
        plot_bias(admission_rates, user_args, axes=subplot_axes[1])
        plt.show()


if __name__ == "__main__":
    main()
