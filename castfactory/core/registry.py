from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, TypeVar


T = TypeVar("T")


class RegistryError(ValueError):
    """Raised when registry lookup or registration fails."""


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, target: Callable[..., T]) -> Callable[..., T]:
        if not name or not isinstance(name, str):
            raise RegistryError(f"{self.kind} registry name must be a non-empty string")
        if name in self._items:
            raise RegistryError(f"{self.kind} '{name}' is already registered")
        self._items[name] = target
        return target

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(
                f"Unknown {self.kind} '{name}'. Available {self.kind}s: {available}"
            ) from exc

    def build(
        self,
        config: Mapping[str, Any],
        *,
        name_key: str = "name",
        extra_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        if name_key not in config:
            raise RegistryError(f"{self.kind} config must contain '{name_key}'")
        name = str(config[name_key])
        kwargs = {key: value for key, value in config.items() if key != name_key}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return self.get(name)(**kwargs)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._items))


readers = Registry("reader")
splitters = Registry("splitter")
representations = Registry("representation")
backbones = Registry("backbone")
bridges = Registry("bridge")
heads = Registry("head")
trainers = Registry("trainer")
parsers = Registry("parser")
metrics = Registry("metric")
rewards = Registry("reward")
objectives = Registry("objective")
protocols = Registry("protocol")


def _register(registry: Registry, name: str, target: Optional[Callable[..., T]]) -> Any:
    if target is None:
        def decorator(obj: Callable[..., T]) -> Callable[..., T]:
            registry.register(name, obj)
            return obj

        return decorator
    registry.register(name, target)
    return target


def register_reader(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(readers, name, target)


def register_splitter(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(splitters, name, target)


def register_representation(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(representations, name, target)


def register_parser(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(parsers, name, target)


def register_backbone(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(backbones, name, target)


def register_bridge(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(bridges, name, target)


def register_head(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(heads, name, target)


def register_trainer(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(trainers, name, target)


def register_metric(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(metrics, name, target)


def register_reward(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(rewards, name, target)


def register_objective(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(objectives, name, target)


def register_protocol(name: str, target: Optional[Callable[..., T]] = None) -> Any:
    return _register(protocols, name, target)


def clear_all() -> None:
    for value in globals().values():
        if isinstance(value, Registry):
            value._items.clear()
