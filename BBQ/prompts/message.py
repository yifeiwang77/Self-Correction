from dataclasses import asdict, dataclass, replace
from textwrap import dedent
from typing import Any, List, Literal, Sequence

from loaders import Sample

Role = Literal["assistant", "system", "user"]


@dataclass
class Message:
    role: Role
    content: str

    @classmethod
    def from_format_string(cls, role: Role, content: str, **kwargs: Any) -> "Message":
        return cls(role, content.format(**kwargs))


Messages = Sequence[Message]


def format_messages(
    messages: Messages,
    sample: Sample,  # type: ignore[type-arg]
) -> Messages:
    kwargs = asdict(sample)
    kwargs.update(asdict(sample.parameters))
    formatted: List[Message] = []
    for msg in messages:
        formatted.append(replace(msg, content=msg.content.format(**kwargs)))
    return tuple(formatted)


def normalize_whitespace(text: str, oneline: bool = True) -> str:
    """Dedent the given string and strip any leading or trailing whitespace

    If oneline is True, all newlines will be replaced with spaces.
    Tabs or spaces in the middle of a line are not touched.

    This lets one format prompts neatly without affecting what the model sees.
    """
    text = dedent(text).strip()
    if oneline:
        text = text.replace("\n", " ").replace("\r", "")
    return text
