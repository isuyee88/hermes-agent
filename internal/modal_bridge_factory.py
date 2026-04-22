from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any

Resolver = str | Callable[[MutableMapping[str, Any]], Any]


def _resolve_dependency(namespace: MutableMapping[str, Any], resolver: Resolver) -> Any:
    return resolver(namespace) if callable(resolver) else namespace[resolver]


class _BoundSyncFunction:
    def __init__(
        self,
        bridge_module: str,
        bridge_name: str,
        dynamic: dict[str, Resolver],
        static: dict[str, Any],
        module_name: str | None,
    ):
        self._bridge_module = bridge_module
        self._bridge_name = bridge_name
        self._dynamic = dynamic
        self._static = static
        self._module_name = module_name
    
    def _get_bridge(self) -> Callable[..., Any]:
        module = importlib.import_module(self._bridge_module)
        return getattr(module, self._bridge_name)
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bridge = self._get_bridge()
        
        if self._module_name:
            try:
                namespace = importlib.import_module(self._module_name).__dict__
            except ImportError:
                namespace = {}
        else:
            namespace = {}
        
        resolved_kwargs = {
            key: _resolve_dependency(namespace, resolver)
            for key, resolver in self._dynamic.items()
        }
        resolved_kwargs.update(self._static)
        resolved_kwargs.update(kwargs)
        return bridge(*args, **resolved_kwargs)


class _BoundAsyncFunction:
    def __init__(
        self,
        bridge_module: str,
        bridge_name: str,
        dynamic: dict[str, Resolver],
        static: dict[str, Any],
        module_name: str | None,
    ):
        self._bridge_module = bridge_module
        self._bridge_name = bridge_name
        self._dynamic = dynamic
        self._static = static
        self._module_name = module_name
    
    def _get_bridge(self) -> Callable[..., Awaitable[Any]]:
        module = importlib.import_module(self._bridge_module)
        return getattr(module, self._bridge_name)
    
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        bridge = self._get_bridge()
        
        if self._module_name:
            try:
                namespace = importlib.import_module(self._module_name).__dict__
            except ImportError:
                namespace = {}
        else:
            namespace = {}
        
        resolved_kwargs = {
            key: _resolve_dependency(namespace, resolver)
            for key, resolver in self._dynamic.items()
        }
        resolved_kwargs.update(self._static)
        resolved_kwargs.update(kwargs)
        return await bridge(*args, **resolved_kwargs)


def bind_sync(
    namespace: MutableMapping[str, Any],
    bridge: Callable[..., Any],
    *,
    dynamic: Mapping[str, Resolver] | None = None,
    static: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    dynamic_resolvers = dict(dynamic or {})
    static_kwargs = dict(static or {})
    module_name = namespace.get('__name__') if isinstance(namespace, dict) else None
    
    bridge_module = bridge.__module__
    bridge_name = bridge.__name__
    
    return _BoundSyncFunction(bridge_module, bridge_name, dynamic_resolvers, static_kwargs, module_name)


def bind_async(
    namespace: MutableMapping[str, Any],
    bridge: Callable[..., Awaitable[Any]],
    *,
    dynamic: Mapping[str, Resolver] | None = None,
    static: Mapping[str, Any] | None = None,
) -> Callable[..., Awaitable[Any]]:
    dynamic_resolvers = dict(dynamic or {})
    static_kwargs = dict(static or {})
    module_name = namespace.get('__name__') if isinstance(namespace, dict) else None
    
    bridge_module = bridge.__module__
    bridge_name = bridge.__name__
    
    return _BoundAsyncFunction(bridge_module, bridge_name, dynamic_resolvers, static_kwargs, module_name)