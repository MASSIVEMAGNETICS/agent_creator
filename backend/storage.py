from backend.models.agent import Agent
from backend.models.deployment import Deployment
from typing import Any

# In-memory storage — thread-safe for single-process uvicorn
agents_db: dict[str, Agent] = {}
deployments_db: dict[str, Deployment] = {}

# Per-agent AdaptiveAgent runtime instances (keyed by agent id)
agent_runtimes: dict[str, Any] = {}
