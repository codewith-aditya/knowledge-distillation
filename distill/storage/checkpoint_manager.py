"""
Checkpoint Manager - Crash recovery system
Saves system state periodically for automatic resume
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class CheckpointManager:
    """
    Manages system checkpoints for crash recovery

    Checkpoint contains:
    - Active topics and their state
    - Questions asked per topic
    - Last completed sample info
    - Queue statistics
    - Worker statistics
    - Timestamp
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.latest_checkpoint_file = self.checkpoint_path / "latest.json"

    async def save_checkpoint(self, state: Dict) -> bool:
        """
        Save system checkpoint

        State should contain:
        {
            "timestamp": str,
            "active_topics": list,
            "total_samples": int,
            "total_questions": int,
            "queue_stats": dict,
            "worker_stats": dict,
            "storage_stats": dict
        }
        """
        try:
            checkpoint_data = {
                **state,
                "checkpoint_time": datetime.utcnow().isoformat() + 'Z'
            }

            temp_file = self.checkpoint_path / f"checkpoint_{int(datetime.utcnow().timestamp())}.json"

            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)

            if self.latest_checkpoint_file.exists():
                self.latest_checkpoint_file.unlink()

            temp_file.rename(self.latest_checkpoint_file)

            return True

        except Exception as e:
            return False

    async def load_checkpoint(self) -> Optional[Dict]:
        """Load latest checkpoint"""
        try:
            if not self.latest_checkpoint_file.exists():
                return None

            with open(self.latest_checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        except Exception:
            return None

    async def has_checkpoint(self) -> bool:
        """Check if checkpoint exists"""
        return self.latest_checkpoint_file.exists()

    def get_checkpoint_age_minutes(self) -> Optional[float]:
        """Get checkpoint age in minutes"""
        try:
            if not self.latest_checkpoint_file.exists():
                return None

            mtime = self.latest_checkpoint_file.stat().st_mtime
            age_seconds = datetime.now().timestamp() - mtime
            return age_seconds / 60

        except Exception:
            return None
