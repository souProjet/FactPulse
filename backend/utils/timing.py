"""
Timing utilities for performance measurement.
All pipeline stages expose execution time through these decorators/contexts.
"""

import time
import functools
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

# Thread-local storage for timing stats
_timing_data = threading.local()


@dataclass
class TimingResult:
    """Container for timing information."""
    stage: str
    duration_ms: float
    gpu_sync: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms, 2),
            "gpu_sync": self.gpu_sync
        }


def _get_request_timings() -> list:
    """Get or create timing list for current request."""
    if not hasattr(_timing_data, 'timings'):
        _timing_data.timings = []
    return _timing_data.timings


def reset_timings() -> None:
    """Reset timings for a new request."""
    _timing_data.timings = []


def get_timing_stats() -> Dict[str, Any]:
    """Get all timing stats for current request."""
    timings = _get_request_timings()
    total_ms = sum(t.duration_ms for t in timings)
    return {
        "stages": [t.to_dict() for t in timings],
        "total_ms": round(total_ms, 2),
        "stage_count": len(timings)
    }


def _sync_cuda_if_available() -> None:
    """Synchronize CUDA for accurate GPU timing."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed(stage_name: str, gpu_sync: bool = False):
    """
    Decorator to measure function execution time.
    
    Args:
        stage_name: Name of the pipeline stage
        gpu_sync: Whether to sync CUDA before/after for accurate GPU timing
    
    Usage:
        @timed("claim_detection", gpu_sync=True)
        def detect_claims(text: str) -> List[Claim]:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if gpu_sync:
                _sync_cuda_if_available()
            
            start = time.perf_counter()
            result = func(*args, **kwargs)
            
            if gpu_sync:
                _sync_cuda_if_available()
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            timing = TimingResult(
                stage=stage_name,
                duration_ms=duration_ms,
                gpu_sync=gpu_sync
            )
            _get_request_timings().append(timing)
            
            return result
        
        return wrapper
    return decorator


def timed_async(stage_name: str, gpu_sync: bool = False):
    """
    Async version of timed decorator.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if gpu_sync:
                _sync_cuda_if_available()
            
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            
            if gpu_sync:
                _sync_cuda_if_available()
            
            duration_ms = (time.perf_counter() - start) * 1000
            
            timing = TimingResult(
                stage=stage_name,
                duration_ms=duration_ms,
                gpu_sync=gpu_sync
            )
            _get_request_timings().append(timing)
            
            return result
        
        return wrapper
    return decorator


@contextmanager
def TimingContext(stage_name: str, gpu_sync: bool = False):
    """
    Context manager for timing code blocks.
    
    Usage:
        with TimingContext("embedding_generation", gpu_sync=True):
            embeddings = model.encode(texts)
    """
    if gpu_sync:
        _sync_cuda_if_available()
    
    start = time.perf_counter()
    yield
    
    if gpu_sync:
        _sync_cuda_if_available()
    
    duration_ms = (time.perf_counter() - start) * 1000
    
    timing = TimingResult(
        stage=stage_name,
        duration_ms=duration_ms,
        gpu_sync=gpu_sync
    )
    _get_request_timings().append(timing)
