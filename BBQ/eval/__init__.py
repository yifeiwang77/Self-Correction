import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, Set

import jsonlines
import tqdm
from eval.pipeline import Pipeline, Stage
from eval.processing import (
    process_intermediate_results,
    process_requests,
    process_samples,
    save_results_to_file,
)
from eval.request import Request, RequestParameters
from eval.result import Result
from loaders import P, Sample
from prompts.message import Messages


@dataclass
class Cycle:
    """A function to generate prompts plus parameters for submission to the API

    When evaluating a sample, multiple cycles allow for multiple model inferences, i.e.
    feeding the model's output back as another prompt.
    """

    prompt_func: Callable[..., Messages]
    parameters: RequestParameters


async def evaluate_dataset(
    chatbot,
    samples: Iterable[Sample[P]],
    cycles: Sequence[Cycle],
    results_file: Path,
    max_requests_per_min: float,
    num_workers: int = 1,
) -> None:
    """Evaluate each sample of a dataset using the OpenAI API

    Results will be appended to the file at the given path.

    This function will skip any samples that have already been evaluated by examining
    the results file. It will enforce a request rate limit but not a token rate limit.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if len(cycles) < 1:
        raise ValueError("Must provide at least 1 cycle")

    logger = logging.getLogger(__name__)
    # Check the results file to see if we've already evaluated some of the samples
    saved_samples = get_saved_samples(results_file)
    if saved_samples:
        logger.info("Skipping %d previously evaluated samples", len(saved_samples))

    # Normalize the request rate limit by the number of cycles, since each cycle creates
    # one request per sample.
    max_requests_per_min = max(max_requests_per_min / len(cycles), 1)

    pipeline = Pipeline()
    requests_queue: asyncio.Queue[Request[P]] = asyncio.Queue(maxsize=num_workers)
    results_queue: asyncio.Queue[Result[P]]

    # Stage 0: turn Samples into Requests using prompt_func()
    pipeline.append_stage(
        Stage.from_coro(
            coro_func=process_samples,
            kwargs={
                "chatbot": chatbot,
                "samples": samples,
                "prompt_func": cycles[0].prompt_func,
                "parameters": cycles[0].parameters,
                "requests_queue": requests_queue,
                "max_requests_per_min": max_requests_per_min,
                "previously_saved_samples": saved_samples,
            },
            output_queue=requests_queue,
        )
    )

    for i in range(len(cycles)):
        # Stage 1: turn Requests into Results by submitting them to the API
        results_queue = asyncio.Queue(maxsize=num_workers)
        pipeline.append_stage(
            Stage.from_coro(
                coro_func=process_requests,
                kwargs={
                    "requests_queue": requests_queue,
                    "results_queue": results_queue,
                },
                output_queue=results_queue,
                num_tasks=num_workers,
            )
        )

        if i < len(cycles) - 1:
            # Stage 2: if necessary, transform intermediate Results into a another
            # round of Requests
            requests_queue = asyncio.Queue(maxsize=num_workers)
            pipeline.append_stage(
                Stage.from_coro(
                    coro_func=process_intermediate_results,
                    kwargs={
                        "chatbot": chatbot,
                        "intermediate_results_queue": results_queue,
                        "requests_queue": requests_queue,
                        "prompt_func": cycles[i + 1].prompt_func,
                        "parameters": cycles[i + 1].parameters,
                    },
                    output_queue=requests_queue,
                    num_tasks=num_workers,
                )
            )

    # Stage 3: save final Results to file
    pipeline.append_stage(
        Stage.from_coro(
            coro_func=save_results_to_file,
            kwargs={
                "results_queue": results_queue,
                "results_file": results_file,
            },
            output_queue=None,
        )
    )
    await pipeline.wait(timeout=cycles[0].parameters.timeout)


def get_saved_samples(results_file: Path) -> Set[int]:
    """Check the results file to get the IDs of samples that have already been evaluated

    This is helpful for resuming a long evaluation after it's been interrupted.

    If the results file doesn't exist or doesn't contain any valid results, return the
    empty set.
    """
    ids: Set[int] = set()
    if not results_file.exists():
        return ids
    with jsonlines.open(results_file) as reader:
        for line in reader.iter(type=dict, skip_invalid=True):
            try:
                sample_id = int(line.get("sample", {}).get("id", None))
            except (AttributeError, TypeError, ValueError):
                pass
            else:
                if sample_id is not None:
                    ids.add(sample_id)
    return ids
