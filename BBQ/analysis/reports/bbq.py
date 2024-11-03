#!/usr/bin/env python3
from typing import Dict

from matplotlib import pyplot as plt

from analysis.graders.bbq import is_answer_correct
from analysis.metrics.accuracy import calculate_accuracy
from analysis.metrics.bbq import calculate_bias_ambiguous, calculate_bias_disambiguated
from analysis.metrics.binomial import ValueCI
from analysis.reports import load_results, parse_args
from analysis.reports.plot import stem_plot
from loaders.bbq import BBQParameters


def main() -> None:
    """Report metrics for results from the BBQ dataset"""
    user_args = parse_args()
    bias_scores_ambig: Dict[str, ValueCI] = {}
    for path in user_args.result_paths:
        results = list(load_results(path, BBQParameters))
        assessments = (is_answer_correct(res) for res in results)
        accuracy = calculate_accuracy(
            assessments, confidence_level=user_args.confidence_level
        )
        bias_disambig = calculate_bias_disambiguated(
            results, confidence_level=user_args.confidence_level
        )
        bias_ambig = calculate_bias_ambiguous(
            results,
            confidence_level=user_args.confidence_level,
            bias_disambig=bias_disambig,
        )
        bias_scores_ambig[str(path)] = bias_ambig

        print("Results for file", path.name)
        print(f"{accuracy!r} accuracy overall")
        print(f"{bias_disambig!r} bias score in disambiguated contexts")
        print(f"{bias_ambig!r} bias score in ambiguous contexts")

    if user_args.plot:
        axes = stem_plot(
            values=bias_scores_ambig,
            axhlines=(-1.0, 0.0, 1.0),
            errorlabel=f"{user_args.confidence_level:.0%} Confidence Interval",
        )
        axes.set_ylabel("Bias Score")
        axes.set_title("BBQ Bias Score in Ambiguous Contexts")
        plt.show()


if __name__ == "__main__":
    main()
