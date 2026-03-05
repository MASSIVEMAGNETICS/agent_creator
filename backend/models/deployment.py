from enum import Enum
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DeploymentEnvironment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SANDBOX = "sandbox"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    FAILED = "failed"
    UNDEPLOYING = "undeploying"
    INACTIVE = "inactive"


class DeploymentConfig(BaseModel):
    agent_id: str
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    replicas: int = Field(default=1, ge=1, le=10)
    auto_scale: bool = False
    max_requests_per_minute: int = 60
    metadata: dict = Field(default_factory=dict)


class Deployment(BaseModel):
    id: str
    agent_id: str
    environment: DeploymentEnvironment
    status: DeploymentStatus = DeploymentStatus.PENDING
    replicas: int = 1
    auto_scale: bool = False
    max_requests_per_minute: int = 60
    endpoint_url: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_at: Optional[datetime] = None
    metadata: dict = Field(default_factory=dict)
