"""
Batch Writer — Efficient batched I/O for high-throughput scenarios.

Buffers samples in memory (configurable batch size, default 10).
Writes to disk in batches with batch-level fsync.

Reduces I/O syscall overhead at scale while maintaining crash safety
(max one batch lost on crash, not one sample).
"""

import os
import asyncio
from pathlib import Path
from typing import Optional
from datetime import datetime
from threading import Lock


class BatchWriter:
    """
    Batched I/O writer for high-throughput sample writing.

    Collects samples in a buffer. Flushes when:
    - Buffer reaches batch_size
    - flush() is called explicitly
    - Time since last flush exceeds max_flush_interval_seconds
    """

    def __init__(
        self,
        file_path: str,
        batch_size: int = 10,
        max_flush_interval_seconds: float = 30.0,
        enable_fsync: bool = True,
    ):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_flush_interval = max_flush_interval_seconds
        self.enable_fsync = enable_fsync

        self._buffer: list[str] = []
        self._lock = Lock()
        self._handle = None
        self._last_flush_time = datetime.utcnow()

        self.total_written = 0
        self.total_flushes = 0

    def add(self, text: str):
        """Add text to the write buffer. Auto-flushes if batch is full."""
        with self._lock:
            self._buffer.append(text)

            if len(self._buffer) >= self.batch_size:
                self._flush_internal()

    def flush(self):
        """Force flush the buffer to disk"""
        with self._lock:
            self._flush_internal()

    def _flush_internal(self):
        """Internal flush — must be called within lock"""
        if not self._buffer:
            return

        if self._handle is None:
            self._handle = open(self.file_path, 'a', encoding='utf-8', buffering=0)

        batch_text = ''.join(self._buffer)
        self._handle.write(batch_text)
        self._handle.flush()

        if self.enable_fsync:
            os.fsync(self._handle.fileno())

        n = len(self._buffer)
        self.total_written += n
        self.total_flushes += 1
        self._buffer.clear()
        self._last_flush_time = datetime.utcnow()

    async def auto_flush_loop(self):
        """Background loop that flushes periodically"""
        while True:
            await asyncio.sleep(self.max_flush_interval)
            with self._lock:
                elapsed = (datetime.utcnow() - self._last_flush_time).total_seconds()
                if elapsed >= self.max_flush_interval and self._buffer:
                    self._flush_internal()

    def get_stats(self) -> dict:
        """Get batch writer statistics"""
        with self._lock:
            return {
                "file": str(self.file_path),
                "buffer_size": len(self._buffer),
                "batch_size": self.batch_size,
                "total_written": self.total_written,
                "total_flushes": self.total_flushes,
                "seconds_since_last_flush": round(
                    (datetime.utcnow() - self._last_flush_time).total_seconds(), 1
                ),
            }

    def close(self):
        """Flush and close"""
        self.flush()
        if self._handle:
            self._handle.close()
            self._handle = None
