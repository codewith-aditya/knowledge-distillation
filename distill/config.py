"""
Hierarchical configuration system
Priority: ENV > config.yaml > defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Config:
    """System configuration optimized for maximum dataset quality"""

    # LLM-1: Research Agent (Question Generator)
    research_provider: str = "openai"
    research_base_url: str = "https://api.openai.com/v1"
    research_model: str = "deepseek-chat"
    research_api_key: str = ""
    research_max_tokens: int = 4096
    research_temperature: float = 0.7

    # LLM-2: Teacher Model (Answer Provider)
    teacher_provider: str = "openai"
    teacher_base_url: str = "https://api.openai.com/v1"
    teacher_model: str = "deepseek-reasoner"
    teacher_api_key: str = ""
    teacher_max_tokens: int = 16000
    teacher_temperature: float = 0.3

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"

    # Storage Paths (Ubuntu VPS paths)
    dataset_base_path: str = "/root/datasets"
    logs_base_path: str = "/root/logs"
    checkpoint_base_path: str = "/root/checkpoints"
    config_base_path: str = "/root/config"

    # RAW TXT Storage Settings
    raw_txt_rotation_size_gb: float = 5.0
    enable_immediate_sync: bool = True

    # Pipeline Settings (Optimized for Quality)
    max_questions_per_topic: int = 200
    difficulty_levels: list = field(default_factory=lambda: ["beginner", "intermediate", "expert", "advanced"])
    min_answer_length: int = 500
    min_quality_score: float = 0.75
    quality_retry_attempts: int = 3

    # Exhaustion Detection
    exhaustion_similarity_threshold: float = 0.88
    exhaustion_min_questions: int = 20

    # Duplicate Detection
    duplicate_exact_threshold: float = 1.0
    duplicate_semantic_threshold: float = 0.92
    embedding_model: str = "text-embedding-3-small"
    embedding_cache_size: int = 10000

    # Workers
    num_topic_workers: int = 1
    num_question_workers: int = 2
    num_answer_workers: int = 3
    num_quality_workers: int = 2

    # Monitoring
    log_level: str = "INFO"
    enable_rich_dashboard: bool = True
    dashboard_refresh_rate: float = 1.0

    # Checkpointing
    checkpoint_interval_minutes: int = 10
    auto_resume_on_start: bool = True

    # Multi-Teacher Settings (Phase 1)
    teachers_config_path: str = "config/teachers.yaml"
    enable_multi_teacher: bool = True
    min_teachers_required: int = 1
    merger_consensus_threshold: int = 2
    teacher_query_timeout: float = 300.0
    teacher_concurrency_limit: int = 6

    # Rate Limiting
    api_request_delay: float = 0.5
    max_retries: int = 5
    retry_base_delay: float = 2.0

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file with env var overrides"""
        config_path = Path(path)

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                raw = f.read()
            # Expand ${VAR} placeholders so config files can stay free of
            # hostnames: endpoints live in .env, not in the repo.
            raw = os.path.expandvars(raw)
            yaml_data = yaml.safe_load(raw) or {}
        else:
            yaml_data = {}

        # Override with environment variables
        env_overrides = {}
        for key in cls.__dataclass_fields__.keys():
            env_key = f"DISTILL_{key.upper()}"
            if env_key in os.environ:
                value = os.environ[env_key]
                field_type = cls.__dataclass_fields__[key].type

                if field_type == bool:
                    env_overrides[key] = value.lower() in ('true', '1', 'yes')
                elif field_type == int:
                    env_overrides[key] = int(value)
                elif field_type == float:
                    env_overrides[key] = float(value)
                else:
                    env_overrides[key] = value

        merged_data = {**yaml_data, **env_overrides}
        return cls(**{k: v for k, v in merged_data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables only"""
        return cls.from_yaml("")

    def to_dict(self) -> dict:
        """Export configuration as dictionary"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}

    def save_yaml(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def ensure_paths(self):
        """Create all necessary directories"""
        for path_attr in ['dataset_base_path', 'logs_base_path', 'checkpoint_base_path', 'config_base_path']:
            path = Path(getattr(self, path_attr))
            path.mkdir(parents=True, exist_ok=True)
