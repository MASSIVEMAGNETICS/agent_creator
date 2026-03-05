from enum import Enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AgentArchetype(str, Enum):
    EXPLORER = "explorer"
    GUARDIAN = "guardian"
    SYNTHESIZER = "synthesizer"
    ORACLE = "oracle"
    EXECUTOR = "executor"
    COLLABORATOR = "collaborator"


class BehaviorProfile(BaseModel):
    curiosity: float = Field(default=0.5, ge=0.0, le=1.0)
    caution: float = Field(default=0.5, ge=0.0, le=1.0)
    creativity: float = Field(default=0.5, ge=0.0, le=1.0)
    precision: float = Field(default=0.5, ge=0.0, le=1.0)
    autonomy: float = Field(default=0.5, ge=0.0, le=1.0)
    empathy: float = Field(default=0.5, ge=0.0, le=1.0)


class AgentConfig(BaseModel):
    name: str
    description: str
    archetype: AgentArchetype
    behavior_profile: BehaviorProfile = Field(default_factory=BehaviorProfile)
    capabilities: list[str] = Field(default_factory=list)
    system_prompt: str = ""
    memory_enabled: bool = True
    can_spawn_children: bool = False
    max_children: int = 0
    ecosystem_tags: list[str] = Field(default_factory=list)


class Agent(BaseModel):
    id: str
    config: AgentConfig
    status: str = "draft"  # "draft", "active", "deployed", "archived"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    evolution_generation: int = 0
    parent_agent_id: Optional[str] = None
    child_agent_ids: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class AgentRunRequest(BaseModel):
    input: str
    context: dict = Field(default_factory=dict)


class AgentAdaptRequest(BaseModel):
    feedback: dict


class AgentSpawnRequest(BaseModel):
    task: str


class AgentResponse(BaseModel):
    agent_id: str
    input: str
    output: str
    intent: Optional[dict] = None
    capabilities_used: list[str] = Field(default_factory=list)
    behavior_influence: dict = Field(default_factory=dict)
    memory_updated: bool = False
    processing_time_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)
