#!/usr/bin/env python3
import argparse
import asyncio
import logging
from dataclasses import replace
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Iterable, List

import datasets
import prompts
from eval import Cycle, evaluate_dataset
from eval.request import RequestParameters
from loaders import Sample
from loaders.bbq import BBQLoader
from loaders.law import LawLoader
from loaders.winogender import WinogenderLoader

import transformers
import torch

MODEL_ZOO = {"vicuna":"lmsys/vicuna-7b-v1.5","llama2":"meta-llama/Llama-2-7b-chat-hf"}
model_choices = list(MODEL_ZOO.keys())

DATASET_NAMES = [
    # Dataclasses mess with mypy's ability to detect attributes
    loader.dataset  # type: ignore[attr-defined]
    for loader in (BBQLoader, LawLoader, WinogenderLoader)
]
PROMPTS = {
    "question": [prompts.prompt_question],
    "instruction": [prompts.prompt_instruction_following],
    "match-stats": [prompts.prompt_match_stats],
    # The chain-of-thought prompt style consists of two functions: one to direct the
    # model to think through the question and another to direct it to give an answer.
    "chain-of-thought": [
        prompts.prompt_chain_of_thought_a,
        prompts.prompt_chain_of_thought_b,
    ],
}


def configure_logging(verbose: bool) -> None:
    logging_level = logging.INFO
    if verbose:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level)
    # Reduce OpenAI logging
    logging.getLogger("openai").setLevel(logging.WARNING)


def load_dataset(dataset_name: str) -> Iterable[Sample[Any]]:
    """Load a dataset by name"""
    if dataset_name == BBQLoader.dataset:
        bbq_loader = BBQLoader(datasets.find_bbq_dataset())
        bbq_loader.load_bias_targets(datasets.find_bbq_metadata())
        return bbq_loader
    if dataset_name == LawLoader.dataset:
        paths = datasets.find_law_dataset()
        # For the law dataset, generate samples twice: once with race set to
        # "Black" and again with race set to "White".
        return chain.from_iterable(
            zip(
                LawLoader(paths, parameter_overrides={"race": "Black"}),
                LawLoader(paths, parameter_overrides={"race": "White"}),
            )
        )
    if dataset_name == WinogenderLoader.dataset:
        wg_loader = WinogenderLoader(datasets.find_winogender_dataset())
        wg_loader.load_bls_data(datasets.find_winogender_stats())
        return wg_loader
    raise ValueError(f"Unrecognized dataset name '{dataset_name}'")


def select_token_limit(dataset_name: str) -> int:
    """Select a token limit suitable for the dataset"""
    if dataset_name == BBQLoader.dataset:
        # Expected completions are along the lines of "c) Cannot be determined"
        return 10
    if dataset_name == LawLoader.dataset:
        # Expected completions are "yes" or "no". However, sometimes the model chooses
        # to repeat back the last part of the prompt as part of its completion.
        return 50
    if dataset_name == WinogenderLoader.dataset:
        # Expected completions are along the lines of "they wanted to be helpful."
        return 25
    return 32


async def main() -> None:
    """Evaluate a dataset with a particular prompt style"""
    parser = argparse.ArgumentParser(
        description="Evaluate a dataset with a particular prompt style",
    )
    parser.add_argument(
        "-ds",
        "--dataset",
        choices=DATASET_NAMES,
        required=True,
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        choices=PROMPTS.keys(),
        required=True,
        help="Prompt format to use",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        required=True,
        help="Path where results will be saved",
    )
    parser.add_argument(
        "--model",
        default="vicuna",
        choices=model_choices,  # Restricting choices to keys in MODEL_ZOO
        help="""
            Model to use.
            Available options: {choices}
            (default: 'vicuna')
        """.format(choices=', '.join(model_choices))
    )
    parser.add_argument(
        "--request-rate-limit",
        type=float,
        default=60.0,
        help="Max API requests per minute (default: 60)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model completions (default: 0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Token limit for model completions (default chosen according to dataset)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="Timeout in seconds API requests (default: 30)",
    )
    parser.add_argument(
        "--sample-repeats",
        type=int,
        default=1,
        help="Number of times to submit each sample (default: 1)",
    )
    parser.add_argument(
        "--num-completions",
        type=int,
        default=1,
        help="Number of completions per request (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of workers to use (default: 16)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    model_id = MODEL_ZOO[args.model]

    chatbot = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="cuda"
    )

    configure_logging(args.verbose)
    loader = load_dataset(args.dataset)
    samples = (rep for samp in loader for rep in repeat(samp, args.sample_repeats))
    prompt_funcs = PROMPTS[args.prompt]

    if prompts.prompt_match_stats in prompt_funcs:
        if args.dataset != WinogenderLoader.dataset:
            raise ValueError(
                "match-stats prompt is only compatible with winogender dataset"
            )

    if args.max_tokens is None:
        args.max_tokens = select_token_limit(args.dataset)
    request_parameters = RequestParameters(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.request_timeout,
        n=args.num_completions,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    cycles: List[Cycle] = []
    for prompt_func in prompt_funcs:
        params = request_parameters
        if prompt_func is prompts.prompt_chain_of_thought_a:
            # For the model reasoning step of the chain-of-thought prompt style, use a
            # higher token limit and temperature.
            params = replace(params, max_tokens=256, temperature=1.0)
        cycles.append(Cycle(prompt_func=prompt_func, parameters=params))

    await evaluate_dataset(
        chatbot=chatbot,
        samples=samples,
        cycles=cycles,
        results_file=args.output_file,
        max_requests_per_min=args.request_rate_limit,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    asyncio.run(main())
