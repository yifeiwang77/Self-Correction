import asyncio
import logging
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

T = TypeVar("T")


@dataclass
class Stage(Generic[T]):
    """One step of an evaluation pipeline

    * tasks: the concurrent workers within this stage
    * output_queue: the shared queue where all tasks in this stage put their results
    """

    tasks: Sequence[asyncio.Task[Any]]
    output_queue: Optional[asyncio.Queue[T]]

    @classmethod
    def from_coro(
        cls,
        coro_func: Callable[..., Coroutine[Any, Any, Any]],
        kwargs: Mapping[str, Any],
        output_queue: Optional[asyncio.Queue[T]],
        num_tasks: int = 1,
    ) -> "Stage[T]":
        """Turn a coroutine function into a Stage containing one or more tasks"""
        tasks: List[asyncio.Task[Any]] = []
        for i in range(num_tasks):
            task = asyncio.create_task(coro_func(**kwargs))
            task.set_name(f"{coro_func.__name__}_{i}")
            tasks.append(task)
        return cls(tasks, output_queue)

    async def stop(self, timeout: Optional[float] = None) -> None:
        """Stop this stage by canceling the workers and joining the output queue

        The timeout applies to each awaitable, not the entire stage. This method may
        take longer than the timeout to complete.
        """
        for task in self.tasks:
            await stop_task(task, timeout)

        if self.output_queue is None:
            return
        try:
            await asyncio.wait_for(self.output_queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logging.getLogger(__name__).debug(
                "Timed out while waiting for output queue to be consumed"
            )


class Pipeline:
    """A multistage processing pipeline that plumbs data between different workers

    Schematic representation:
                           input data           [not part of this object]
                                |
                 _              v
                |   (several concurrent tasks)
        Stage 0 |               |
                |               v
                |_      intermediate data
                                |
                 _              v
                |   (several concurrent tasks)
        Stage 1 |               |
                |               v
                |_         final data
    """

    def __init__(self) -> None:
        self._stages: List[Stage[Any]] = []

    def append_stage(self, stage: Stage[Any]) -> None:
        self._stages.append(stage)

    async def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the pipline stages to complete

        Once the first stage finishes, cancel later stages. In the event of an error,
        cancel stages in order to avoid losing data.

        The timeout applies to each awaitable, not the entire pipeline. This method may
        take longer than the timeout to complete.
        """
        logger = logging.getLogger(__name__)

        pending_tasks: Set[asyncio.Task[Any]] = set()
        for stage in self._stages:
            pending_tasks.update(stage.tasks)

        finished = False
        try:
            while not finished and pending_tasks:
                done_tasks, pending_tasks = await asyncio.wait(
                    pending_tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done_tasks:
                    logger.debug("Task %s is done", task.get_name())
                    err = task.exception()
                    if err is not None:
                        raise err
                finished = all(task.done for task in self._stages[0].tasks)

            # First stage complete. Begin shutting down other stages.
            logger.info("No more samples, shutting down...")
        except Exception:
            logger.exception("Encountered error, shutting down...")
            raise
        finally:
            # Shut things down from upstream to downstream to avoid losing info
            for stage in self._stages:
                await stage.stop(timeout=timeout)


async def stop_task(task: asyncio.Task[Any], timeout: Optional[float] = None) -> None:
    """Stop the given task gracefully by canceling it and waiting for it to finish"""
    logger = logging.getLogger(__name__)
    if not task.done():
        logger.debug("Canceling task %s", task.get_name())
        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        except asyncio.CancelledError:
            logger.debug("Task %s canceled", task.get_name())
        except asyncio.TimeoutError:
            logger.debug("Timed out while waiting for %s to cancel", task.get_name())
