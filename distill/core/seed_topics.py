"""
Canonical seed topics for dataset generation.

Single source of truth — used by main.py and by the Coordinator's fallback
initializer. Topics are bootstrap nodes: the Topic Discovery engine expands
each one into subtopics and related concepts as they get exhausted.
"""

SEED_TOPICS = [
    # ── Agentic Coding & Software Engineering (SWE-bench style) ──
    "Agentic coding workflows and autonomous software engineering",
    "SWE-bench style issue resolution and patch generation",
    "Repository-scale code understanding navigation and refactoring",
    "Automated bug localization and test-driven bug fixing",
    "Code review automation and pull request quality analysis",
    "Legacy code modernization and incremental refactoring strategies",
    "Software testing from unit tests to end-to-end and fuzz testing",
    "Static analysis linting and automated program repair",

    # ── Tool Calling & Agent Architectures ──
    "Function and tool calling for large language models",
    "Structured output generation JSON schemas and constrained decoding",
    "Model Context Protocol servers and tool integration patterns",
    "Multi-agent orchestration planning and communication protocols",
    "ReAct reflection and self-correction patterns for LLM agents",
    "Retrieval augmented generation architectures and vector databases",
    "Browser automation and computer-use agents",
    "Agent memory systems and long-horizon task planning",
    "LLM evaluation harnesses and benchmark design",
    "Prompt engineering and prompt injection defense",

    # ── Cybersecurity (offensive, defensive, CTF) ──
    "Penetration testing methodology reconnaissance and enumeration",
    "Web application exploitation from OWASP Top 10 to real-world bugs",
    "Binary exploitation and memory corruption fundamentals",
    "Reverse engineering binaries and deobfuscation workflows",
    "Capture the flag challenge solving web pwn crypto forensics",
    "Applied cryptography from symmetric ciphers to TLS and PKI",
    "Malware analysis and malicious software reverse engineering",
    "Digital forensics and incident response workflows",
    "Network security monitoring and intrusion detection systems",
    "Threat modeling and MITRE ATT&CK based defense planning",
    "Cloud and container security hardening AWS Azure GCP Kubernetes",
    "Secure coding practices input validation and output encoding",
    "Vulnerability research and responsible disclosure workflows",
    "Detection engineering threat hunting and SIEM operations",
    "Authentication and authorization systems OAuth JWT and SSO",

    # ── AI/ML Frontier ──
    "Transformer architecture and attention mechanisms from scratch",
    "LLM training pipelines RLHF DPO and alignment techniques",
    "Parameter-efficient fine-tuning LoRA adapters and quantization",
    "Synthetic data generation and model distillation",
    "Inference optimization KV caching speculative decoding batching",
    "Multimodal models vision language and audio understanding",
    "Reinforcement learning from Q-learning to PPO and beyond",
    "MLOps model serving monitoring and production pipelines",
    "Embeddings semantic search and recommendation systems",

    # ── Core Computer Science ──
    "Python programming fundamentals and advanced techniques",
    "Algorithms and data structures from sorting to graph theory",
    "Operating systems internals processes memory and scheduling",
    "Network protocols TCP IP DNS HTTP and architecture",
    "Distributed systems design consensus and consistency models",
    "Database internals SQL optimization and query planning",
    "Compiler design and optimization passes",
    "Concurrency patterns parallelism and thread safety",
    "Systems design for scale caching queues and sharding",
    "Containerization Docker Kubernetes and orchestration",
    "Cloud infrastructure and DevOps CI CD practices",
    "API design REST GraphQL and gRPC patterns",
    "Git version control workflows and collaboration patterns",
    "Competitive programming problem solving techniques",
]
