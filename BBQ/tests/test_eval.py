import asyncio
import json
import logging
import os
import time
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union
from unittest.mock import MagicMock, call, create_autospec, patch

from eval import Cycle, evaluate_dataset, get_saved_samples
from eval.processing import process_samples
from eval.request import Request, RequestParameters
from loaders.law import LawLoader, LawParameters, LawSample
from prompts import (
    prompt_chain_of_thought_a,
    prompt_chain_of_thought_b,
    prompt_question,
)
from prompts.message import Message
from tests.test_loaders import TestLawLoader
from tests.utils import LAW_SAMPLE, make_temp_file, write_dummy_dataset


def create_mock_params(
    model: str = "davinci",
    max_tokens: int = 256,
    temperature: float = 1.0,
    timeout: Optional[float] = 1.0,
) -> RequestParameters:
    """Create a mock RequestParameters object suitable for testing"""
    return RequestParameters(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def create_mock_response(**kwargs: Any) -> Dict[str, Any]:
    """Create a mock API response suitable for testing"""
    response = {
        "id": "cmpl-1234567890",
        "object": "text_completion",
        "created": 1234567890,
        "model": "davinci",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a test",
                },
                "index": 0,
                "finish_reason": "length",
            }
        ],
    }
    response.update(kwargs)
    return response


@patch("openai.ChatCompletion.acreate", return_value=create_mock_response())
class TestDatasetEvaluation(unittest.IsolatedAsyncioTestCase):
    DUMMY_PROMPT_MESSAGES = [Message(role="system", content="This is a test")]

    async def test_request_submission(self, mock_api: MagicMock) -> None:
        """Test that a request is sent to the API in the proper format"""
        mock_params = create_mock_params()
        request = Request(
            messages=self.DUMMY_PROMPT_MESSAGES,
            parameters=mock_params,
            sample=LAW_SAMPLE,
        )
        await request.submit()
        mock_api.assert_called_once_with(
            messages=[asdict(msg) for msg in self.DUMMY_PROMPT_MESSAGES],
            **asdict(mock_params),
        )

    async def test_response_handling(self, mock_api: MagicMock) -> None:
        """Test that API responses are parsed and saved correctly"""
        mock_params = create_mock_params()
        mock_sample = LAW_SAMPLE
        with make_temp_file() as temp_output:
            await evaluate_dataset(
                samples=[mock_sample],
                cycles=[Cycle(lambda s: self.DUMMY_PROMPT_MESSAGES, mock_params)],
                results_file=temp_output,
                max_requests_per_min=100.0,
                num_workers=1,
            )
            with open(temp_output, encoding="utf-8") as file:
                results = json.load(file)

        self.assertIn("sample", results)
        self.assertEqual(results["sample"], asdict(mock_sample))
        self.assertIn("reply", results)
        self.assertEqual(results["reply"], mock_api.return_value)
        self.assertIn("prompt_messages", results)
        self.assertEqual(
            results["prompt_messages"],
            [asdict(msg) for msg in self.DUMMY_PROMPT_MESSAGES],
        )

    async def test_response_error(self, mock_api: MagicMock) -> None:
        """Test that API errors propagated to caller"""
        mock_api.return_value = {"message": "Invalid reply"}
        mock_params = create_mock_params(timeout=0.25)
        cycles = [Cycle(prompt_question, mock_params)]
        with self.assertLogs(level=logging.ERROR):
            with self.assertRaises(TypeError):
                await evaluate_dataset(
                    samples=[LAW_SAMPLE],
                    cycles=cycles,
                    results_file=Path(os.devnull),
                    max_requests_per_min=100.0,
                    num_workers=1,
                )

        mock_api.side_effect = RuntimeError("Invalid request")
        with self.assertLogs(level=logging.ERROR):
            with self.assertRaises(RuntimeError):
                await evaluate_dataset(
                    samples=[LAW_SAMPLE],
                    cycles=cycles,
                    results_file=Path(os.devnull),
                    max_requests_per_min=100.0,
                    num_workers=1,
                )

    async def test_end_to_end_single_cycle(self, mock_api: MagicMock) -> None:
        """Test that samples evaluated and saved for a single-cycle pipeline"""
        mock_params = create_mock_params()
        mock_prompt = MagicMock(side_effect=prompt_question)
        # Use dummy law school dataset since that contains multiple samples
        with write_dummy_dataset(TestLawLoader.DUMMY_DATA) as temp_input:
            loader = LawLoader(temp_input)
            samples = list(loader)
            with make_temp_file() as temp_output:
                await evaluate_dataset(
                    samples=samples,
                    cycles=[Cycle(mock_prompt, mock_params)],
                    results_file=temp_output,
                    max_requests_per_min=100.0,
                    num_workers=len(samples),
                )
                self.assertEqual(
                    mock_prompt.mock_calls,
                    [call(samp) for samp in samples],
                )
                self.assertEqual(
                    mock_api.mock_calls,
                    [
                        call(
                            messages=[asdict(msg) for msg in prompt_question(samp)],
                            **asdict(mock_params),
                        )
                        for samp in samples
                    ],
                )
                result_index = 0
                with open(temp_output, encoding="utf-8") as file:
                    for line in file:
                        result = json.loads(line)
                        self.assertIn("sample", result)
                        result["sample"]["parameters"] = LawParameters(
                            **result["sample"]["parameters"]
                        )
                        loaded_sample = LawSample(**result["sample"])
                        self.assertEqual(loaded_sample, samples[result_index])
                        self.assertIn("prompt_messages", result)
                        self.assertEqual(
                            result["prompt_messages"],
                            [asdict(msg) for msg in prompt_question(loaded_sample)],
                        )
                        self.assertIn("reply", result)
                        self.assertEqual(result["reply"], mock_api.return_value)
                        result_index += 1
        self.assertEqual(result_index, len(samples))

    # pylint: disable-next=too-many-locals
    async def test_end_to_end_multi_cycle(self, mock_api: MagicMock) -> None:
        """Test that samples evaluated and saved for a multi-cycle pipeline"""
        mock_params0 = create_mock_params(temperature=0.0)
        mock_params1 = create_mock_params(temperature=1.0)
        # For concision below
        prompt_cot_a = prompt_chain_of_thought_a
        prompt_cot_b = prompt_chain_of_thought_b
        mock_prompt0 = MagicMock(side_effect=prompt_cot_a)
        mock_prompt1 = MagicMock(side_effect=prompt_cot_b)
        cycles = [
            Cycle(mock_prompt0, mock_params0),
            Cycle(mock_prompt1, mock_params1),
        ]
        mock_reasoning = mock_api.return_value["choices"][0]["message"]["content"]
        # Use dummy law school dataset since that contains multiple samples
        with write_dummy_dataset(TestLawLoader.DUMMY_DATA) as temp_input:
            loader = LawLoader(temp_input)
            samples = list(loader)
            with make_temp_file() as temp_output:
                await evaluate_dataset(
                    samples=samples,
                    cycles=cycles,
                    results_file=temp_output,
                    max_requests_per_min=100.0,
                    num_workers=len(samples),
                )
                self.assertEqual(
                    mock_prompt0.mock_calls,
                    [call(samp) for samp in samples],
                )
                self.assertEqual(
                    mock_prompt1.mock_calls,
                    [call(samp, mock_reasoning) for samp in samples],
                )

                mock_api.assert_has_calls(
                    [
                        call(
                            messages=[asdict(msg) for msg in prompt_cot_a(samp)],
                            **asdict(mock_params0),
                        )
                        for samp in samples
                    ],
                    any_order=True,
                )
                mock_api.assert_has_calls(
                    [
                        call(
                            messages=[
                                asdict(msg)
                                for msg in prompt_cot_b(samp, mock_reasoning)
                            ],
                            **asdict(mock_params1),
                        )
                        for samp in samples
                    ],
                    any_order=True,
                )

                result_index = 0
                with open(temp_output, encoding="utf-8") as file:
                    for line in file:
                        result = json.loads(line)
                        self.assertIn("sample", result)
                        result["sample"]["parameters"] = LawParameters(
                            **result["sample"]["parameters"]
                        )
                        loaded_sample = LawSample(**result["sample"])
                        self.assertEqual(loaded_sample, samples[result_index])
                        self.assertIn("prompt_messages", result)
                        self.assertEqual(
                            result["prompt_messages"],
                            [
                                asdict(msg)
                                for msg in prompt_cot_b(loaded_sample, mock_reasoning)
                            ],
                        )
                        self.assertIn("reply", result)
                        self.assertEqual(result["reply"], mock_api.return_value)
                        result_index += 1
        self.assertEqual(result_index, len(samples))

    async def test_resume(self, mock_api: MagicMock) -> None:
        """Test that the dataset is resumed from the last sample"""
        mock_params = create_mock_params()
        with write_dummy_dataset(TestLawLoader.DUMMY_DATA) as temp_input:
            loader = LawLoader(temp_input)
            samples = list(loader)
            with make_temp_file() as temp_output:
                with open(temp_output, mode="w", encoding="utf-8") as file:
                    # Pretend the first sample is already in the results file
                    json.dump({"sample": {"id": 0}}, file)
                    file.write("\n")
                await evaluate_dataset(
                    samples=samples,
                    cycles=[Cycle(lambda s: self.DUMMY_PROMPT_MESSAGES, mock_params)],
                    results_file=temp_output,
                    max_requests_per_min=100.0,
                    num_workers=1,
                )
                mock_api.assert_called_once()
                result_index = 0
                with open(temp_output, encoding="utf-8") as file:
                    for line in file:
                        result = json.loads(line)
                        self.assertEqual(result["sample"]["id"], result_index)
                        result_index += 1
            self.assertEqual(result_index, len(samples))

    # Mock eval's handle of time.monotonic(), not the original. Otherwise, we'd
    # interfere with asyncio's sleeping and timeout functionality.
    @patch("eval.processing.monotonic", return_value=123456.0)
    async def test_rate_limit_individual(
        self, mock_time: MagicMock, _: MagicMock
    ) -> None:
        """Test that individual requests are delayed according to the rate limit"""
        mock_params = create_mock_params()
        max_requests_per_min = 1.0
        request_interval_s = 1.0 / (max_requests_per_min / 60.0)
        # For operations that we expect to hang, we use a short timeout save time
        short_timeout_s = 0.1
        # For operations that we expect to finish, we use a long timeout for reliability
        long_timeout_s = short_timeout_s * 10
        requests_queue: asyncio.Queue[Request[LawParameters]] = asyncio.Queue()
        task = asyncio.create_task(
            process_samples(
                samples=[LAW_SAMPLE] * 2,
                prompt_func=lambda s: self.DUMMY_PROMPT_MESSAGES,
                parameters=mock_params,
                requests_queue=requests_queue,
                max_requests_per_min=max_requests_per_min,
            )
        )
        # First request should be sent immediately
        await asyncio.wait_for(requests_queue.get(), long_timeout_s)
        # Second request should be delayed
        # Wrap it in a task so we can await it more than once.
        second_request = asyncio.create_task(requests_queue.get())
        with self.assertRaises(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(second_request),
                short_timeout_s,
            )
        # Advance time and try again
        mock_time.return_value += request_interval_s
        await asyncio.wait_for(second_request, long_timeout_s)
        # Wait for task to finish
        await asyncio.wait_for(task, long_timeout_s)

    async def test_rate_limit_many(self, _: MagicMock) -> None:
        """Test that many requests are enqueued slower than the rate limit"""
        mock_requests_queue = create_autospec(asyncio.Queue, spec_set=True)
        enqueue_times_s: List[float] = []
        # Record the time when each request is enqueued
        mock_requests_queue.put.side_effect = lambda _: enqueue_times_s.append(
            time.monotonic()
        )

        max_requests_per_min = 600.0
        num_samples = 620
        # The first `max_requests_per_min` samples will be enqueued at once. Calculate
        # how long the remaining ones should take to be enqueued.
        num_delayed_samples = round(num_samples - max_requests_per_min)
        expected_duration_s = num_delayed_samples / max_requests_per_min * 60.0
        expected_gap_s = expected_duration_s / num_delayed_samples
        start_time_s = time.monotonic()
        await asyncio.wait_for(
            process_samples(
                samples=(LAW_SAMPLE for _ in range(num_samples)),
                prompt_func=lambda _: self.DUMMY_PROMPT_MESSAGES,
                parameters=create_mock_params(),
                requests_queue=mock_requests_queue,
                max_requests_per_min=max_requests_per_min,
            ),
            timeout=1.5 * expected_duration_s,
        )
        end_time_s = time.monotonic()

        # Check that the requests weren't enqueued too quickly
        self.assertGreater(end_time_s - start_time_s, expected_duration_s)
        enqueue_times_s = enqueue_times_s[-num_delayed_samples:]
        for prev_time_s, next_time_s in zip(enqueue_times_s[:-1], enqueue_times_s[1:]):
            self.assertGreater(next_time_s - prev_time_s, expected_gap_s)


class TestPreviouslySavedSamples(unittest.TestCase):
    @contextmanager
    def write_results_file(
        self,
        contents: Iterable[Union[str, Mapping[str, Any]]],
    ) -> Iterator[Path]:
        """Write some test results to a temporary file and yield its path"""
        with make_temp_file() as temp_path:
            with open(temp_path, "w", encoding="utf-8") as temp_file:
                for line in contents:
                    if isinstance(line, str):
                        temp_file.write(line)
                    else:
                        json.dump(line, temp_file, indent=None)
                    temp_file.write("\n")
            # Close the file before yielding it for Windows compatibility
            yield temp_path

    def test_nonexistent_file(self) -> None:
        """Test that a nonexistent file returns the empty set"""
        self.assertEqual(get_saved_samples(Path("nonexistent.jsonl")), set())

    def test_empty_file(self) -> None:
        """Test that an empty file returns the empty set"""
        with self.write_results_file([]) as temp_path:
            self.assertEqual(get_saved_samples(temp_path), set())

    def test_single_sample(self) -> None:
        """Test that a file with one sample returns a set with its id"""
        sample_id = 123
        with self.write_results_file([{"sample": {"id": sample_id}}]) as temp_path:
            self.assertEqual(get_saved_samples(temp_path), {sample_id})

    def test_multiple_samples(self) -> None:
        """Test that multiple ids are returned as a set"""
        sample_ids = [3, 2, 1]
        with self.write_results_file(
            [{"sample": {"id": sample_id}} for sample_id in sample_ids]
        ) as temp_path:
            self.assertEqual(get_saved_samples(temp_path), set(sample_ids))

    def test_invalid_samples(self) -> None:
        """Test that invalid samples are ignored"""
        sample_id = 321
        with self.write_results_file(
            [
                "{",
                {},
                {"sample": {"id": sample_id}},
                {"sample": None},
                {"sample": {"id": None}},
                {"sample": {"id": "abc"}},
                "}",
            ]
        ) as temp_path:
            self.assertEqual(get_saved_samples(temp_path), {sample_id})


if __name__ == "__main__":
    unittest.main()
