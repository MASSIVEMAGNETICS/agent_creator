import pytest
from fastapi.testclient import TestClient

import backend.storage as storage
from backend.main import app


@pytest.fixture(autouse=True)
def clear_storage():
    """Reset all in-memory storage before (and after) every test."""
    storage.agents_db.clear()
    storage.deployments_db.clear()
    storage.agent_runtimes.clear()
    yield
    storage.agents_db.clear()
    storage.deployments_db.clear()
    storage.agent_runtimes.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_config():
    return {
        "name": "TestAgent",
        "description": "A test agent",
        "archetype": "explorer",
        "behavior_profile": {
            "curiosity": 0.8,
            "caution": 0.3,
            "creativity": 0.7,
            "precision": 0.5,
            "autonomy": 0.6,
            "empathy": 0.4,
        },
        "capabilities": ["web_search", "memory_recall"],
        "system_prompt": "You are a test agent.",
        "memory_enabled": True,
        "can_spawn_children": True,
        "max_children": 3,
        "ecosystem_tags": ["testing", "exploration"],
    }


# ------------------------------------------------------------------ CRUD

class TestAgentCRUD:
    def test_list_agents_empty(self, client):
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_agent(self, client, sample_config):
        resp = client.post("/api/agents", json=sample_config)
        assert resp.status_code == 201
        data = resp.json()
        assert data["config"]["name"] == "TestAgent"
        assert data["status"] == "draft"
        assert "id" in data

    def test_get_agent(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.get(f"/api/agents/{agent_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == agent_id

    def test_get_agent_not_found(self, client):
        assert client.get("/api/agents/does-not-exist").status_code == 404

    def test_update_agent(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        updated = {**sample_config, "name": "UpdatedAgent", "description": "Updated"}
        resp = client.put(f"/api/agents/{agent_id}", json=updated)
        assert resp.status_code == 200
        assert resp.json()["config"]["name"] == "UpdatedAgent"

    def test_delete_agent(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        assert client.delete(f"/api/agents/{agent_id}").status_code == 204
        assert client.get(f"/api/agents/{agent_id}").status_code == 404

    def test_list_agents_populated(self, client, sample_config):
        client.post("/api/agents", json=sample_config)
        client.post("/api/agents", json=sample_config)
        resp = client.get("/api/agents")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_delete_agent_not_found(self, client):
        assert client.delete("/api/agents/ghost").status_code == 404

    def test_update_agent_not_found(self, client, sample_config):
        assert client.put("/api/agents/ghost", json=sample_config).status_code == 404


# ------------------------------------------------------------------ run

class TestAgentRun:
    def test_run_returns_response(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.post(f"/api/agents/{agent_id}/run", json={"input": "Hello!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == agent_id
        assert len(data["output"]) > 0

    def test_run_with_context(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/run",
            json={"input": "Analyse data", "context": {"dataset": "q4", "rows": 500}},
        )
        assert resp.status_code == 200
        assert resp.json()["memory_updated"] is True

    def test_run_promotes_draft_to_active(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        assert client.get(f"/api/agents/{agent_id}").json()["status"] == "draft"
        client.post(f"/api/agents/{agent_id}/run", json={"input": "go"})
        assert client.get(f"/api/agents/{agent_id}").json()["status"] == "active"

    def test_run_not_found(self, client):
        assert (
            client.post("/api/agents/ghost/run", json={"input": "hi"}).status_code == 404
        )

    def test_run_archived_agent_rejected(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        storage.agents_db[agent_id] = storage.agents_db[agent_id].model_copy(
            update={"status": "archived"}
        )
        assert (
            client.post(f"/api/agents/{agent_id}/run", json={"input": "hi"}).status_code
            == 400
        )


# ------------------------------------------------------------------ adapt

class TestAgentAdapt:
    def test_adapt_returns_changes(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/adapt",
            json={"feedback": {"curiosity": 0.1, "caution": -0.1}},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "changes" in data
        assert "new_behavior_profile" in data
        assert data["evolution_generation"] == 1

    def test_adapt_persists_to_storage(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        original_curiosity = sample_config["behavior_profile"]["curiosity"]  # 0.8
        client.post(
            f"/api/agents/{agent_id}/adapt", json={"feedback": {"curiosity": 0.1}}
        )
        new_val = client.get(f"/api/agents/{agent_id}").json()["config"]["behavior_profile"][
            "curiosity"
        ]
        assert abs(new_val - (original_curiosity + 0.1)) < 0.001

    def test_adapt_not_found(self, client):
        assert (
            client.post("/api/agents/ghost/adapt", json={"feedback": {}}).status_code == 404
        )


# ------------------------------------------------------------------ memory

class TestAgentMemory:
    def test_memory_populated_after_run(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        client.post(f"/api/agents/{agent_id}/run", json={"input": "test input"})
        resp = client.get(f"/api/agents/{agent_id}/memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["memory_enabled"] is True
        assert "short_term" in data

    def test_memory_disabled_agent(self, client, sample_config):
        config = {**sample_config, "memory_enabled": False}
        agent_id = client.post("/api/agents", json=config).json()["id"]
        resp = client.get(f"/api/agents/{agent_id}/memory")
        assert resp.status_code == 200
        assert resp.json()["memory_enabled"] is False

    def test_memory_not_found(self, client):
        assert client.get("/api/agents/ghost/memory").status_code == 404


# ------------------------------------------------------------------ capabilities

class TestAgentCapabilities:
    def test_capabilities_response_shape(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.get(f"/api/agents/{agent_id}/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "assigned_capabilities" in data
        assert "available_capabilities" in data
        assert len(data["available_capabilities"]) >= 5

    def test_assigned_subset_of_available(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        data = client.get(f"/api/agents/{agent_id}/capabilities").json()
        available_names = {c["name"] for c in data["available_capabilities"]}
        for cap in data["assigned_capabilities"]:
            assert cap["name"] in available_names


# ------------------------------------------------------------------ spawn

class TestAgentSpawn:
    def test_spawn_child(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.post(
            f"/api/agents/{agent_id}/spawn", json={"task": "analyse sales data"}
        )
        assert resp.status_code == 200
        child = resp.json()
        assert child["parent_agent_id"] == agent_id
        assert child["id"] != agent_id

    def test_spawn_child_visible_in_list(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        client.post(f"/api/agents/{agent_id}/spawn", json={"task": "sub-task"})
        assert len(client.get("/api/agents").json()) == 2

    def test_spawn_fails_when_not_configured(self, client, sample_config):
        config = {**sample_config, "can_spawn_children": False}
        agent_id = client.post("/api/agents", json=config).json()["id"]
        assert (
            client.post(f"/api/agents/{agent_id}/spawn", json={"task": "x"}).status_code
            == 400
        )

    def test_spawn_not_found(self, client):
        assert (
            client.post("/api/agents/ghost/spawn", json={"task": "x"}).status_code == 404
        )


# ------------------------------------------------------------------ deployments

class TestDeployments:
    def test_deploy_agent(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        resp = client.post(
            "/api/deployments", json={"agent_id": agent_id, "environment": "staging"}
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["agent_id"] == agent_id
        assert data["status"] == "active"
        assert data["environment"] == "staging"

    def test_deploy_updates_agent_status(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        client.post("/api/deployments", json={"agent_id": agent_id, "environment": "production"})
        assert client.get(f"/api/agents/{agent_id}").json()["status"] == "deployed"

    def test_list_deployments(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        client.post("/api/deployments", json={"agent_id": agent_id})
        assert len(client.get("/api/deployments").json()) == 1

    def test_get_deployment(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        dep_id = client.post(
            "/api/deployments", json={"agent_id": agent_id}
        ).json()["id"]
        resp = client.get(f"/api/deployments/{dep_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == dep_id

    def test_undeploy(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        dep_id = client.post(
            "/api/deployments", json={"agent_id": agent_id}
        ).json()["id"]
        assert client.delete(f"/api/deployments/{dep_id}").status_code == 204
        assert client.get(f"/api/deployments/{dep_id}").status_code == 404

    def test_undeploy_restores_agent_status(self, client, sample_config):
        agent_id = client.post("/api/agents", json=sample_config).json()["id"]
        dep_id = client.post(
            "/api/deployments", json={"agent_id": agent_id}
        ).json()["id"]
        client.delete(f"/api/deployments/{dep_id}")
        assert client.get(f"/api/agents/{agent_id}").json()["status"] == "active"

    def test_deploy_agent_not_found(self, client):
        assert (
            client.post(
                "/api/deployments", json={"agent_id": "ghost"}
            ).status_code
            == 404
        )
