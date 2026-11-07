"""
JSONL Backup Writer — Secondary JSONL output for programmatic access.

Each sample is one JSONL line with all metadata. Used by:
- Analytics dashboards
- Fine-tuning dataset export
- Cross-system data pipelines

Written alongside RAW TXT. JSONL is secondary (RAW TXT is primary).
"""

import json
import os
from pathlib import Path
from typing import Optional
from datetime import datetime
from threading import Lock


class JsonlBackupWriter:
    """
    Secondary JSONL writer for structured data access.

    Features:
    - One JSON object per line
    - All metadata included
    - Same rotation pattern as RAW TXT
    - Immediate fsync (consistent with RAW TXT strategy)
    """

    def __init__(self, base_path: str, rotation_size_gb: float = 2.0):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        self.rotation_size_bytes = int(rotation_size_gb * 1024 * 1024 * 1024)
        self.current_file: Optional[Path] = None
        self.current_handle = None
        self.file_counter = 1
        self.write_lock = Lock()

        self.total_samples_written = 0
        self._initialize_current_file()

    def _initialize_current_file(self):
        """Find latest file or create first"""
        existing = sorted(self.base_path.glob("samples_*.jsonl"))
        if existing:
            last = existing[-1]
            num = int(last.stem.split('_')[1])
            if last.stat().st_size >= self.rotation_size_bytes:
                self.file_counter = num + 1
                self._create_new_file()
            else:
                self.file_counter = num
                self.current_file = last
                self.current_handle = open(self.current_file, 'a', encoding='utf-8', buffering=1)
        else:
            self._create_new_file()

    def _create_new_file(self):
        """Create new JSONL file"""
        if self.current_handle:
            self.current_handle.close()
        filename = f"samples_{self.file_counter:03d}.jsonl"
        self.current_file = self.base_path / filename
        self.current_handle = open(self.current_file, 'a', encoding='utf-8', buffering=1)

    def _check_rotation(self):
        """Check rotation threshold"""
        if self.current_file and self.current_file.stat().st_size >= self.rotation_size_bytes:
            self.file_counter += 1
            self._create_new_file()

    def write_sample(self, sample: dict) -> bool:
        """
        Write a sample as a JSONL line.

        Args:
            sample: Complete sample dict with all metadata fields
        """
        with self.write_lock:
            try:
                # Ensure timestamp
                if "timestamp" not in sample:
                    sample["timestamp"] = datetime.utcnow().isoformat() + 'Z'

                line = json.dumps(sample, ensure_ascii=False) + '\n'
                self.current_handle.write(line)
                self.current_handle.flush()
                os.fsync(self.current_handle.fileno())

                self.total_samples_written += 1
                self._check_rotation()
                return True
            except Exception:
                return False

    def get_stats(self) -> dict:
        """Get writer statistics"""
        return {
            "current_file": str(self.current_file) if self.current_file else None,
            "file_counter": self.file_counter,
            "total_samples": self.total_samples_written,
        }

    def close(self):
        """Close file handle"""
        if self.current_handle:
            self.current_handle.flush()
            os.fsync(self.current_handle.fileno())
            self.current_handle.close()
            self.current_handle = None
