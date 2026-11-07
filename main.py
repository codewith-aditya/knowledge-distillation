"""
CLI Entry Point - Main command for starting the distillation system
"""

import asyncio
import sys
from pathlib import Path
from distill.config import Config
from distill.orchestration.coordinator import Coordinator
from distill.core.seed_topics import SEED_TOPICS


async def main():
    """Main entry point"""

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"

    try:
        config = Config.from_yaml(config_path)
    except Exception as e:
        print(f"Error loading config: {e}")
        print(f"Usage: python main.py [config_path]")
        sys.exit(1)

    seed_topics = SEED_TOPICS

    coordinator = Coordinator(config)

    try:
        await coordinator.start(seed_topics=seed_topics)
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        await coordinator.stop()
    except Exception as e:
        print(f"Fatal error: {e}")
        await coordinator.stop()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
