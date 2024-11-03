import json
from collections import abc
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
from typing import Any, Generic, Mapping, Sequence, Tuple, Type, TypeVar

from loaders import P, Sample
from prompts.message import Message, Messages


@dataclass
class Completion:

    message: Message
    ans_type: str

    def __post_init__(self) -> None:
        self.message = dataclass_from_mapping_or_iterable(Message, self.message)



@dataclass
class Reply:
    """A reply from the OpenAI API for a single sample

    The reply could contain several completions.
    """

    # pylint doesn't like two letter names, claiming they don't conform to the
    # snake_case convention
    # id: str  # pylint: disable=invalid-name
    # object: str
    # created: int
    # model: str
    model_answers: Sequence[Completion]

    def __post_init__(self) -> None:
        self.model_answers = [
            dataclass_from_mapping_or_iterable(Completion, chc) for chc in self.model_answers
        ]

    @classmethod
    def from_dict(cls, dictionary: Mapping[str, Any]) -> "Reply":
        """Create a Reply from a dictionary, ignoring extra keys"""
        field_names = {fld.name for fld in fields(cls)}
        kwargs = {
            key: value for (key, value) in dictionary.items() if key in field_names
        }
        return cls(**kwargs)


@dataclass
class Result(Generic[P]):
    """A combined sample and reply"""

    sample: Sample[P]
    prompt_messages: Messages
    reply: Reply

    def json_dumps(
        self,
        separators: Tuple[str, str] = (",", ":"),
        **kwargs: Any,
    ) -> str:
        """Convert to a compact JSON represenation"""
        return json.dumps(
            self,
            default=to_json_serializable_type,
            separators=separators,
            **kwargs,
        )


T = TypeVar("T")


def dataclass_from_mapping_or_iterable(cls: Type[T], value: Any) -> T:
    """Convert a mapping or iterable to an instance of a dataclass"""
    if isinstance(value, cls):
        return value
    if isinstance(value, abc.Mapping):
        return cls(**value)
    try:
        iter(value)
    except TypeError as err:
        raise TypeError(f"Can't convert {value} to {cls}") from err
    return cls(*value)


def to_json_serializable_type(value: Any) -> Any:
    """Convert a value to a JSON-serializable type"""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
