import logging
import time
import os
import tracemalloc

def get_logger(name: str) -> logging.Logger:
    """Returns a configured industrial logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class Profiler:
    """Context manager for profiling execution time and memory usage."""
    def __init__(self, name: str, logger: logging.Logger = None):
        self.name = name
        self.logger = logger or get_logger(self.name)
        self.start_time = 0
        self.end_time = 0
        self.mem_start = 0
        self.mem_end = 0

    def __enter__(self):
        tracemalloc.start()
        self.mem_start = tracemalloc.get_traced_memory()[0]
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.mem_end = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        elapsed_ms = (self.end_time - self.start_time) * 1000
        mem_diff_kb = (self.mem_end - self.mem_start) / 1024
        
        self.logger.info(f"[{self.name}] Executed in {elapsed_ms:.2f} ms | Mem Diff: {mem_diff_kb:.2f} KB")
        self.elapsed_ms = elapsed_ms
        self.mem_diff_kb = mem_diff_kb
