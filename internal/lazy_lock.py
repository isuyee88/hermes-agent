"""
Lazy thread-safe lock wrapper to avoid serialization issues.

Module-level threading.Lock() instances cannot be serialized by Modal's
deployment system. This wrapper creates locks lazily on first access.
"""
import threading
from typing import Any

class LazyLock:
    """A thread lock that is created lazily on first use.
    
    This avoids the 'cannot pickle _thread.lock' error during Modal deployment
    because the lock object doesn't exist until it's actually needed.
    """
    
    _instance = None
    _lock = threading.Lock()
    _locks: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def _get_lock(self) -> threading.Lock:
        """Get or create the actual lock."""
        import inspect
        # Use caller's module name as key to get unique locks per module
        frame = inspect.currentframe()
        if frame and frame.f_back:
            key = frame.f_back.f_globals.get('__name__', 'default')
        else:
            key = 'default'
        
        if key not in self._locks:
            with self._lock:
                if key not in self._locks:
                    self._locks[key] = threading.Lock()
        return self._locks[key]
    
    def acquire(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_lock().acquire(*args, **kwargs)
    
    def release(self, *args: Any, **kwargs: Any) -> Any:
        return self._get_lock().release(*args, **kwargs)
    
    def __enter__(self) -> 'LazyLock':
        self._get_lock().__enter__()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self._get_lock().__exit__(*args)
    
    def locked(self) -> bool:
        return self._get_lock().locked()


def create_lazy_lock() -> LazyLock:
    """Create a lazy lock that will be initialized on first use."""
    return LazyLock()
