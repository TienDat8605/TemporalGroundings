"""Explicit registries: easy to extend, easy to inspect, no import magic."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
Factory = Callable[..., T]


class Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Factory[Any]] = {}

    def register(self, name: str, factory: Factory[Any]) -> None:
        if name in self._factories:
            raise ValueError(f"duplicate {self.kind}: {name}")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> Any:
        try:
            factory = self._factories[name]
        except KeyError as error:
            choices = ", ".join(self.names()) or "none"
            raise ValueError(f"unknown {self.kind} {name!r}; choose from {choices}") from error
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


METHODS: Registry = Registry("method")
MODELS: Registry = Registry("model")
BENCHMARKS: Registry = Registry("benchmark")


def load_builtin_plugins() -> None:
    # Imports are intentionally local so `hybrid-vtg --help` stays lightweight.
    from .benchmarks import register_benchmarks
    from .methods import register_methods
    from .models import register_models

    if not BENCHMARKS.names():
        register_benchmarks(BENCHMARKS)
    if not METHODS.names():
        register_methods(METHODS)
    if not MODELS.names():
        register_models(MODELS)
