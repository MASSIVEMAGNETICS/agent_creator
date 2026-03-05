import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

import backend.storage as storage
from backend.models.deployment import Deployment, DeploymentEnvironment, DeploymentStatus

router = APIRouter(prefix="/api/deployments", tags=["deployments"])


class DeployRequest(BaseModel):
    agent_id: str
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT
    replicas: int = 1
    auto_scale: bool = False
    max_requests_per_minute: int = 60
    metadata: dict = {}


@router.post("", response_model=Deployment, status_code=status.HTTP_201_CREATED)
async def deploy_agent(request: DeployRequest) -> Deployment:
    agent = storage.agents_db.get(request.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {request.agent_id} not found")

    deployment_id = str(uuid.uuid4())
    now = datetime.utcnow()

    deployment = Deployment(
        id=deployment_id,
        agent_id=request.agent_id,
        environment=request.environment,
        status=DeploymentStatus.ACTIVE,
        replicas=request.replicas,
        auto_scale=request.auto_scale,
        max_requests_per_minute=request.max_requests_per_minute,
        endpoint_url=f"/api/agents/{request.agent_id}/run",
        created_at=now,
        updated_at=now,
        deployed_at=now,
        metadata=request.metadata,
    )
    storage.deployments_db[deployment_id] = deployment

    # Promote agent status to deployed
    storage.agents_db[request.agent_id] = agent.model_copy(
        update={"status": "deployed", "updated_at": now}
    )

    return deployment


@router.get("", response_model=list[Deployment])
async def list_deployments() -> list[Deployment]:
    return list(storage.deployments_db.values())


@router.get("/{deployment_id}", response_model=Deployment)
async def get_deployment(deployment_id: str) -> Deployment:
    deployment = storage.deployments_db.get(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")
    return deployment


@router.delete("/{deployment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def undeploy(deployment_id: str) -> None:
    deployment = storage.deployments_db.get(deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail=f"Deployment {deployment_id} not found")

    storage.deployments_db.pop(deployment_id)

    agent = storage.agents_db.get(deployment.agent_id)
    if agent and agent.status == "deployed":
        storage.agents_db[deployment.agent_id] = agent.model_copy(
            update={"status": "active", "updated_at": datetime.utcnow()}
        )
