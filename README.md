# ⬡ Agent Creator — Next-Generation AI Agent Studio

Build, evolve, and deploy a new paradigm of AI agent. Not a GPT wrapper.
Not another LangChain clone. Something that feels teleported from 2035:
**alive, adaptive, and ecosystem-native**.

---

## ✨ Key Concepts

| Concept | Description |
|---|---|
| **Adaptive Behavior Engine** | Agents have mutable `behavior_profile` weights (curiosity, caution, creativity…) that evolve through feedback. |
| **Intent Graph** | Parses free-form input into structured `Intent` objects with archetype, entities, and sub-task decomposition — no LLM required. |
| **Dynamic Capability Registry** | Named capabilities (tools) are registered and discovered via keyword-based semantic matching. |
| **Living Memory** | Three-tier memory: short-term (last 50 items), long-term (consolidated facts), and episodic (complete interaction records). |
| **Multi-Agent Orchestration** | Agents can `spawn_child()` to delegate specialised sub-tasks, forming adaptive agent trees. |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd agent_creator
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn backend.main:app --reload
```

Open **http://localhost:8000** to launch the Agent Studio UI.

The interactive API docs are available at **http://localhost:8000/docs**.

---

## 🗂 Architecture

```
agent_creator/
├── requirements.txt
├── backend/
│   ├── main.py                   # FastAPI app, CORS, static file mounting
│   ├── storage.py                # In-memory agent & deployment stores
│   ├── models/
│   │   ├── agent.py              # Agent, AgentConfig, BehaviorProfile, AgentResponse
│   │   └── deployment.py        # Deployment, DeploymentConfig, DeploymentStatus
│   ├── core/
│   │   ├── adaptive_agent.py    # AdaptiveAgent runtime — the "new kinda agent"
│   │   ├── capability_registry.py  # Dynamic capability discovery
│   │   ├── intent_graph.py      # Semantic intent extraction & decomposition
│   │   └── agent_memory.py      # Living three-tier memory system
│   └── api/
│       ├── agents.py            # /api/agents CRUD + run/adapt/spawn endpoints
│       └── deployments.py      # /api/deployments deploy/undeploy endpoints
├── frontend/
│   ├── index.html               # Agent Studio single-page app
│   ├── app.js                   # Frontend logic (vanilla JS)
│   └── styles.css               # Dark/futuristic theme
└── tests/
    ├── test_agents_api.py
    ├── test_adaptive_agent.py
    ├── test_intent_graph.py
    └── test_capability_registry.py
```

---

## 🔌 API Reference

### Agents — `/api/agents`

| Method | Path | Description |
|---|---|---|
| GET | `/api/agents` | List all agents |
| POST | `/api/agents` | Create agent |
| GET | `/api/agents/{id}` | Get agent |
| PUT | `/api/agents/{id}` | Update agent config |
| DELETE | `/api/agents/{id}` | Delete agent |
| POST | `/api/agents/{id}/run` | Run agent with `{input, context}` |
| POST | `/api/agents/{id}/adapt` | Adapt behavior with `{feedback}` |
| GET | `/api/agents/{id}/memory` | Inspect agent memory |
| GET | `/api/agents/{id}/capabilities` | List capabilities |
| POST | `/api/agents/{id}/spawn` | Spawn child agent |

### Deployments — `/api/deployments`

| Method | Path | Description |
|---|---|---|
| POST | `/api/deployments` | Deploy an agent |
| GET | `/api/deployments` | List deployments |
| GET | `/api/deployments/{id}` | Get deployment |
| DELETE | `/api/deployments/{id}` | Undeploy |

---

## 🧪 Running Tests

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

No API keys. No external services. Fully self-contained.

---

## Agent Archetypes

| Archetype | Personality |
|---|---|
| 🧭 **Explorer** | High curiosity, discovers possibilities |
| 🛡 **Guardian** | High caution, risk-aware decision making |
| ⚗️ **Synthesizer** | High creativity, integrates cross-domain patterns |
| 🔮 **Oracle** | High precision, predictive analysis |
| ⚡ **Executor** | High autonomy, action-oriented |
| 🤝 **Collaborator** | High empathy, coordination-focused |
A next gen applifcation that creates edits and deploys a new kinda agent 
