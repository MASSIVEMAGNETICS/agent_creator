import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

import backend.storage as storage
from backend.core.adaptive_agent import AdaptiveAgent
from backend.core.capability_registry import get_global_registry
from backend.models.agent import (
    Agent,
    AgentAdaptRequest,
    AgentConfig,
    AgentResponse,
    AgentRunRequest,
    AgentSpawnRequest,
    BehaviorProfile,
)

router = APIRouter(prefix="/api/agents", tags=["agents"])


def _get_or_create_runtime(agent: Agent) -> AdaptiveAgent:
    """Return the cached AdaptiveAgent runtime, creating it on first access."""
    if agent.id not in storage.agent_runtimes:
        storage.agent_runtimes[agent.id] = AdaptiveAgent(
            agent_id=agent.id,
            config=agent.config,
            registry=get_global_registry(),
        )
    return storage.agent_runtimes[agent.id]


# ------------------------------------------------------------------ CRUD

@router.get("", response_model=list[Agent])
async def list_agents() -> list[Agent]:
    return list(storage.agents_db.values())


@router.post("", response_model=Agent, status_code=status.HTTP_201_CREATED)
async def create_agent(config: AgentConfig) -> Agent:
    agent_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    agent = Agent(
        id=agent_id,
        config=config,
        status="draft",
        created_at=now,
        updated_at=now,
    )
    storage.agents_db[agent_id] = agent
    return agent


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(agent_id: str) -> Agent:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return agent


@router.put("/{agent_id}", response_model=Agent)
async def update_agent(agent_id: str, config: AgentConfig) -> Agent:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    # Invalidate runtime so it is re-created with the new config
    storage.agent_runtimes.pop(agent_id, None)

    updated = agent.model_copy(update={"config": config, "updated_at": datetime.now(timezone.utc)})
    storage.agents_db[agent_id] = updated
    return updated


@router.delete("/{agent_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent(agent_id: str) -> None:
    if agent_id not in storage.agents_db:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    storage.agents_db.pop(agent_id)
    storage.agent_runtimes.pop(agent_id, None)


# ------------------------------------------------------------------ run / adapt / memory / capabilities / spawn

@router.post("/{agent_id}/run", response_model=AgentResponse)
async def run_agent(agent_id: str, request: AgentRunRequest) -> AgentResponse:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    if agent.status == "archived":
        raise HTTPException(status_code=400, detail="Cannot run an archived agent")

    runtime = _get_or_create_runtime(agent)

    # Promote draft → active on first run
    if agent.status == "draft":
        storage.agents_db[agent_id] = agent.model_copy(
            update={"status": "active", "updated_at": datetime.now(timezone.utc)}
        )

    return runtime.run(input=request.input, context=request.context)


@router.post("/{agent_id}/adapt", response_model=dict)
async def adapt_agent(agent_id: str, request: AgentAdaptRequest) -> dict:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    runtime = _get_or_create_runtime(agent)
    changes = runtime.adapt(feedback=request.feedback)

    # Sync evolved behavior profile back into persistent storage
    updated_config = agent.config.model_copy(
        update={"behavior_profile": BehaviorProfile(**runtime.behavior_profile)}
    )
    storage.agents_db[agent_id] = agent.model_copy(
        update={
            "config": updated_config,
            "updated_at": datetime.now(timezone.utc),
            "evolution_generation": runtime._generation,
        }
    )

    return {
        "agent_id": agent_id,
        "changes": changes,
        "new_behavior_profile": runtime.behavior_profile,
        "evolution_generation": runtime._generation,
    }


@router.get("/{agent_id}/memory", response_model=dict)
async def get_agent_memory(agent_id: str) -> dict:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    runtime = _get_or_create_runtime(agent)

    if runtime.memory is None:
        return {"memory_enabled": False, "message": "Memory is disabled for this agent"}

    return {"memory_enabled": True, "agent_id": agent_id, **runtime.memory.get_state()}


@router.get("/{agent_id}/capabilities", response_model=dict)
async def get_agent_capabilities(agent_id: str) -> dict:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    registry = get_global_registry()
    all_caps = registry.list_all()
    assigned = set(agent.config.capabilities)

    def _cap_dict(cap) -> dict:
        return {
            "name": cap.name,
            "description": cap.description,
            "tags": cap.tags,
            "usage_count": cap.usage_count,
        }

    return {
        "agent_id": agent_id,
        "assigned_capabilities": [_cap_dict(c) for c in all_caps if c.name in assigned],
        "available_capabilities": [_cap_dict(c) for c in all_caps],
    }


@router.post("/{agent_id}/spawn", response_model=Agent)
async def spawn_child_agent(agent_id: str, request: AgentSpawnRequest) -> Agent:
    agent = storage.agents_db.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    if not agent.config.can_spawn_children:
        raise HTTPException(
            status_code=400, detail="This agent is not configured to spawn children"
        )

    runtime = _get_or_create_runtime(agent)

    try:
        child_runtime = runtime.spawn_child(task=request.task)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    now = datetime.now(timezone.utc)
    child_agent = Agent(
        id=child_runtime.agent_id,
        config=child_runtime.config,
        status="active",
        created_at=now,
        updated_at=now,
        parent_agent_id=agent_id,
    )
    storage.agents_db[child_runtime.agent_id] = child_agent
    storage.agent_runtimes[child_runtime.agent_id] = child_runtime

    # Record child reference on parent
    storage.agents_db[agent_id] = agent.model_copy(
        update={
            "child_agent_ids": agent.child_agent_ids + [child_runtime.agent_id],
            "updated_at": now,
        }
    )

    return child_agent
