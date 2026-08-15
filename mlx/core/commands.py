from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar, runtime_checkable


ResultT = TypeVar("ResultT")


@runtime_checkable
class Command(Protocol, Generic[ResultT]):
    """Small, portable application boundary used by every MLX workflow."""

    def execute(self) -> ResultT:
        ...


@dataclass(frozen=True)
class WorkflowEvent:
    level: str
    message: str
    current: Optional[int] = None
    total: Optional[int] = None
    payload: Any = None


@runtime_checkable
class WorkflowReporter(Protocol):
    """Receives workflow status without coupling commands to a terminal UI."""

    def emit(self, event: WorkflowEvent) -> None:
        ...


class NullWorkflowReporter:
    def emit(self, event: WorkflowEvent) -> None:
        return None


class CallbackWorkflowReporter:
    def __init__(self, callback: Callable[[WorkflowEvent], None]) -> None:
        self.callback = callback

    def emit(self, event: WorkflowEvent) -> None:
        self.callback(event)


def emit(
    reporter: WorkflowReporter,
    level: str,
    message: str,
    *,
    current: Optional[int] = None,
    total: Optional[int] = None,
    payload: Any = None,
) -> None:
    reporter.emit(
        WorkflowEvent(
            level=level,
            message=message,
            current=current,
            total=total,
            payload=payload,
        )
    )
