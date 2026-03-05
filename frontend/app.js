/* ================================================================
   AGENT CREATOR — frontend logic
   Pure JS, no build tools, no external dependencies.
   All API calls go to the FastAPI backend at relative /api/* paths.
================================================================ */

const API = '';  // same origin

// ── State ─────────────────────────────────────────────────────────
let agents        = [];
let currentAgent  = null;
let currentDeploy = null;

const TRAITS = ['curiosity', 'caution', 'creativity', 'precision', 'autonomy', 'empathy'];
const CAPS   = ['web_search', 'code_execute', 'memory_recall', 'send_message', 'spawn_agent'];

// ── Toast ─────────────────────────────────────────────────────────
function toast(msg, error = false) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show' + (error ? ' error' : '');
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.className = ''; }, 3000);
}

// ── API helpers ───────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  if (res.status === 204) return null;
  return res.json();
}

// ── Sidebar helpers ───────────────────────────────────────────────
function archetypeClass(a) {
  return 'a-' + (a || 'explorer');
}
function statusClass(s) {
  return 's-' + (s || 'draft');
}

function renderSidebar() {
  const list  = document.getElementById('agent-list');
  const empty = document.getElementById('sidebar-empty');

  // Remove existing cards (but keep the empty state element)
  [...list.querySelectorAll('.agent-card')].forEach(c => c.remove());

  if (!agents.length) {
    empty.style.display = 'flex';
    return;
  }
  empty.style.display = 'none';

  agents.forEach(agent => {
    const card = document.createElement('div');
    card.className = 'agent-card' + (currentAgent?.id === agent.id ? ' active' : '');
    card.dataset.id = agent.id;
    card.innerHTML = `
      <div class="card-top">
        <span class="card-name">${esc(agent.config.name)}</span>
        <span class="card-status-dot ${statusClass(agent.status)}"></span>
      </div>
      <div class="card-meta">
        <span class="archetype-badge ${archetypeClass(agent.config.archetype)}">${agent.config.archetype}</span>
        <span class="gen-badge">gen·${agent.evolution_generation}</span>
      </div>`;
    card.addEventListener('click', () => selectAgent(agent.id));
    list.appendChild(card);
  });
}

// ── Load agents ───────────────────────────────────────────────────
async function loadAgents() {
  try {
    agents = await api('GET', '/api/agents');
    renderSidebar();
    if (currentAgent) {
      const fresh = agents.find(a => a.id === currentAgent.id);
      if (fresh) renderAgentPanel(fresh);
    }
  } catch (e) {
    toast('Failed to load agents: ' + e.message, true);
  }
}

// ── Select / render agent panel ───────────────────────────────────
function selectAgent(id) {
  const agent = agents.find(a => a.id === id);
  if (!agent) return;
  currentAgent = agent;
  currentDeploy = null;
  renderSidebar();
  renderAgentPanel(agent);
}

function renderAgentPanel(agent) {
  document.getElementById('main-empty').classList.add('hidden');
  document.getElementById('agent-panel').classList.remove('hidden');

  document.getElementById('panel-agent-name').textContent = agent.config.name;
  document.getElementById('panel-agent-id').textContent   = agent.id;

  renderDesignTab(agent);
  renderBehaviorTab(agent);
  loadMemory();
  renderDeployTab(agent);
}

// ── DESIGN tab ────────────────────────────────────────────────────
function renderDesignTab(agent) {
  document.getElementById('design-name').value        = agent.config.name;
  document.getElementById('design-description').value = agent.config.description;
  document.getElementById('design-archetype').value   = agent.config.archetype;
  document.getElementById('design-system-prompt').value = agent.config.system_prompt || '';
  document.getElementById('design-tags').value        = (agent.config.ecosystem_tags || []).join(', ');
  document.getElementById('design-can-spawn').checked = agent.config.can_spawn_children;
  document.getElementById('design-max-children').value = agent.config.max_children || 0;
  document.getElementById('design-status').textContent = '';
  renderCapGrid(agent.config.capabilities || []);
}

function renderCapGrid(selected) {
  const grid = document.getElementById('cap-grid');
  grid.innerHTML = '';
  CAPS.forEach(cap => {
    const checked = selected.includes(cap);
    const label = document.createElement('label');
    label.className = 'cap-check' + (checked ? ' checked' : '');
    label.innerHTML = `<input type="checkbox" value="${cap}" ${checked ? 'checked' : ''} />
      ${capLabel(cap)}`;
    label.querySelector('input').addEventListener('change', e => {
      label.classList.toggle('checked', e.target.checked);
    });
    grid.appendChild(label);
  });
}

function capLabel(cap) {
  const icons = {
    web_search:    '🔍 web_search',
    code_execute:  '⚡ code_execute',
    memory_recall: '🧠 memory_recall',
    send_message:  '✉️ send_message',
    spawn_agent:   '🧬 spawn_agent',
  };
  return icons[cap] || cap;
}

function collectDesignConfig() {
  const caps = [...document.querySelectorAll('#cap-grid input:checked')].map(i => i.value);
  const tags = document.getElementById('design-tags').value
    .split(',').map(t => t.trim()).filter(Boolean);
  const bp = {};
  TRAITS.forEach(t => {
    bp[t] = parseFloat(document.getElementById(`slider-${t}`)?.value ?? 0.5);
  });

  return {
    name:              document.getElementById('design-name').value.trim(),
    description:       document.getElementById('design-description').value.trim(),
    archetype:         document.getElementById('design-archetype').value,
    system_prompt:     document.getElementById('design-system-prompt').value.trim(),
    ecosystem_tags:    tags,
    capabilities:      caps,
    can_spawn_children: document.getElementById('design-can-spawn').checked,
    max_children:      parseInt(document.getElementById('design-max-children').value) || 0,
    memory_enabled:    true,
    behavior_profile:  bp,
  };
}

document.getElementById('btn-save-design').addEventListener('click', async () => {
  if (!currentAgent) return;
  const config = collectDesignConfig();
  if (!config.name) { toast('Agent name is required', true); return; }
  try {
    const updated = await api('PUT', `/api/agents/${currentAgent.id}`, config);
    currentAgent = updated;
    agents = agents.map(a => a.id === updated.id ? updated : a);
    renderSidebar();
    document.getElementById('design-status').textContent = '✓ Saved';
    toast('Agent saved');
  } catch (e) {
    toast('Save failed: ' + e.message, true);
  }
});

// ── BEHAVIOR tab ──────────────────────────────────────────────────
function renderBehaviorTab(agent) {
  document.getElementById('beh-gen').textContent = agent.evolution_generation || 0;
  renderSliders(agent.config.behavior_profile);

  // Show/hide spawn panel
  const showSpawn = agent.config.can_spawn_children;
  document.getElementById('spawn-panel').style.display = showSpawn ? '' : 'none';
}

function renderSliders(bp) {
  const container = document.getElementById('trait-sliders');
  container.innerHTML = '';
  TRAITS.forEach(trait => {
    const val = bp ? (bp[trait] ?? 0.5) : 0.5;
    const row = document.createElement('div');
    row.className = 'trait-row';
    row.innerHTML = `
      <div class="trait-header">
        <span class="trait-name">${trait}</span>
        <span class="trait-val" id="tv-${trait}">${val.toFixed(2)}</span>
      </div>
      <input type="range" id="slider-${trait}" min="0" max="1" step="0.01"
             value="${val}" />`;
    row.querySelector('input').addEventListener('input', e => {
      document.getElementById(`tv-${trait}`).textContent = parseFloat(e.target.value).toFixed(2);
    });
    container.appendChild(row);
  });
}

// Run agent
document.getElementById('btn-run').addEventListener('click', async () => {
  if (!currentAgent) return;
  const input = document.getElementById('run-input').value.trim();
  if (!input) { toast('Enter an instruction', true); return; }

  const btn = document.getElementById('btn-run');
  btn.innerHTML = '<span class="spinner"></span>';
  btn.disabled = true;

  try {
    const resp = await api('POST', `/api/agents/${currentAgent.id}/run`, {
      input,
      context: {},
    });
    const out = document.getElementById('run-output');
    out.textContent = resp.output;
    out.classList.add('visible');

    const meta = document.getElementById('run-meta');
    meta.classList.remove('hidden');
    meta.innerHTML = '';
    if (resp.intent?.archetype) {
      meta.innerHTML += `<span class="meta-chip">intent: ${resp.intent.archetype}</span>`;
    }
    if (resp.intent?.confidence !== undefined) {
      meta.innerHTML += `<span class="meta-chip">conf: ${(resp.intent.confidence * 100).toFixed(0)}%</span>`;
    }
    (resp.capabilities_used || []).forEach(c => {
      meta.innerHTML += `<span class="meta-chip">⚡ ${c}</span>`;
    });
    meta.innerHTML += `<span class="meta-chip">⏱ ${resp.processing_time_ms.toFixed(1)}ms</span>`;

    // Refresh memory after run
    if (document.getElementById('tab-memory').classList.contains('active')) {
      await loadMemory();
    }
  } catch (e) {
    toast('Run failed: ' + e.message, true);
  } finally {
    btn.textContent = 'Run';
    btn.disabled = false;
  }
});

// Allow Enter key in run input
document.getElementById('run-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-run').click();
});

// Adapt buttons
document.querySelectorAll('[data-feedback]').forEach(btn => {
  btn.addEventListener('click', async () => {
    if (!currentAgent) return;
    const feedback = JSON.parse(btn.dataset.feedback);
    try {
      const resp = await api('POST', `/api/agents/${currentAgent.id}/adapt`, { feedback });
      // Sync sliders
      const bp = resp.new_behavior_profile;
      TRAITS.forEach(t => {
        const slider = document.getElementById(`slider-${t}`);
        const label  = document.getElementById(`tv-${t}`);
        if (slider && bp[t] !== undefined) {
          slider.value = bp[t];
          label.textContent = bp[t].toFixed(2);
        }
      });
      document.getElementById('beh-gen').textContent = resp.evolution_generation;
      document.getElementById('adapt-msg').textContent =
        `✓ Adapted — generation ${resp.evolution_generation}`;
      // Update local state
      currentAgent = await api('GET', `/api/agents/${currentAgent.id}`);
      agents = agents.map(a => a.id === currentAgent.id ? currentAgent : a);
      renderSidebar();
    } catch (e) {
      toast('Adapt failed: ' + e.message, true);
    }
  });
});

// Spawn child agent
document.getElementById('btn-spawn').addEventListener('click', async () => {
  if (!currentAgent) return;
  const task = document.getElementById('spawn-task').value.trim();
  if (!task) { toast('Enter a task for the child agent', true); return; }

  const btn = document.getElementById('btn-spawn');
  btn.disabled = true; btn.textContent = '…';

  try {
    const child = await api('POST', `/api/agents/${currentAgent.id}/spawn`, { task });
    document.getElementById('spawn-msg').textContent =
      `✓ Child spawned: ${child.config.name} (${child.id.slice(0, 8)})`;
    document.getElementById('spawn-task').value = '';
    await loadAgents();
    toast(`Child agent "${child.config.name}" created`);
  } catch (e) {
    toast('Spawn failed: ' + e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = 'Spawn';
  }
});

// ── MEMORY tab ────────────────────────────────────────────────────
async function loadMemory() {
  if (!currentAgent) return;
  try {
    const state = await api('GET', `/api/agents/${currentAgent.id}/memory`);
    if (!state.memory_enabled) {
      ['st-memory','lt-memory','ep-memory'].forEach(id =>
        document.getElementById(id).innerHTML =
          '<div class="memory-empty">Memory disabled for this agent</div>'
      );
      return;
    }
    renderMemorySection('st-memory', 'st-count', state.short_term, 'short-term');
    renderLongTermSection('lt-memory', 'lt-count', state.long_term || {});
    renderEpisodicSection('ep-memory', 'ep-count', state.episodic || []);
  } catch (e) {
    toast('Memory load failed: ' + e.message, true);
  }
}

function renderMemorySection(elId, countId, items, type) {
  const el = document.getElementById(elId);
  document.getElementById(countId).textContent = `(${(items||[]).length})`;
  if (!items || !items.length) {
    el.innerHTML = '<div class="memory-empty">No items yet</div>';
    return;
  }
  el.innerHTML = `<table class="memory-table">
    <thead><tr><th>KEY</th><th>VALUE</th><th>HITS</th></tr></thead>
    <tbody>${items.map(i => `
      <tr>
        <td class="mem-key">${esc(i.key)}</td>
        <td class="mem-val">${esc(String(i.value).slice(0, 120))}</td>
        <td class="mem-count">${i.access_count ?? 0}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

function renderLongTermSection(elId, countId, items) {
  const el = document.getElementById(elId);
  const entries = Object.entries(items || {});
  document.getElementById(countId).textContent = `(${entries.length})`;
  if (!entries.length) {
    el.innerHTML = '<div class="memory-empty">No long-term facts consolidated yet</div>';
    return;
  }
  el.innerHTML = `<table class="memory-table">
    <thead><tr><th>KEY</th><th>VALUE</th><th>IMPORTANCE</th></tr></thead>
    <tbody>${entries.map(([k, v]) => `
      <tr>
        <td class="mem-key">${esc(k)}</td>
        <td class="mem-val">${esc(String(v.value).slice(0, 120))}</td>
        <td class="mem-count">${(v.importance ?? 0.5).toFixed(2)}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

function renderEpisodicSection(elId, countId, episodes) {
  const el = document.getElementById(elId);
  document.getElementById(countId).textContent = `(${episodes.length})`;
  if (!episodes.length) {
    el.innerHTML = '<div class="memory-empty">No episodes recorded yet</div>';
    return;
  }
  el.innerHTML = `<table class="memory-table">
    <thead><tr><th>ID</th><th>SUMMARY</th><th>TAGS</th></tr></thead>
    <tbody>${episodes.map(ep => `
      <tr>
        <td class="mem-key">${esc(ep.episode_id)}</td>
        <td class="mem-val">${esc(ep.summary.slice(0, 100))}</td>
        <td class="mem-count text-muted" style="font-size:.62rem">${(ep.tags||[]).join(', ')}</td>
      </tr>`).join('')}
    </tbody></table>`;
}

document.getElementById('btn-refresh-memory').addEventListener('click', loadMemory);

// ── DEPLOY tab ────────────────────────────────────────────────────
async function renderDeployTab(agent) {
  // Try to find existing deployment for this agent
  try {
    const deps = await api('GET', '/api/deployments');
    currentDeploy = deps.find(d => d.agent_id === agent.id) || null;
  } catch (_) {
    currentDeploy = null;
  }
  updateDeployUI();
}

function updateDeployUI() {
  const box = document.getElementById('deploy-status-box');
  const btnDeploy   = document.getElementById('btn-deploy');
  const btnUndeploy = document.getElementById('btn-undeploy');

  if (currentDeploy) {
    box.classList.remove('hidden');
    btnDeploy.classList.add('hidden');
    btnUndeploy.classList.remove('hidden');
    document.getElementById('dep-id').textContent       = currentDeploy.id.slice(0, 16) + '…';
    document.getElementById('dep-status').textContent   = currentDeploy.status.toUpperCase();
    document.getElementById('dep-env').textContent      = currentDeploy.environment;
    document.getElementById('dep-replicas').textContent = currentDeploy.replicas;
    document.getElementById('dep-time').textContent     =
      currentDeploy.deployed_at ? new Date(currentDeploy.deployed_at).toLocaleString() : '—';
    document.getElementById('dep-endpoint').textContent = currentDeploy.endpoint_url || '—';
  } else {
    box.classList.add('hidden');
    btnDeploy.classList.remove('hidden');
    btnUndeploy.classList.add('hidden');
  }
}

document.getElementById('btn-deploy').addEventListener('click', async () => {
  if (!currentAgent) return;
  const env      = document.getElementById('deploy-env').value;
  const replicas = parseInt(document.getElementById('deploy-replicas').value) || 1;
  const rpm      = parseInt(document.getElementById('deploy-rpm').value) || 60;

  const btn = document.getElementById('btn-deploy');
  btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Deploying…';

  try {
    currentDeploy = await api('POST', '/api/deployments', {
      agent_id: currentAgent.id,
      environment: env,
      replicas,
      max_requests_per_minute: rpm,
    });
    updateDeployUI();
    await loadAgents();
    toast(`Agent deployed to ${env}`);
  } catch (e) {
    toast('Deploy failed: ' + e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = '⚡ Deploy';
  }
});

document.getElementById('btn-undeploy').addEventListener('click', async () => {
  if (!currentDeploy) return;
  const btn = document.getElementById('btn-undeploy');
  btn.disabled = true;
  try {
    await api('DELETE', `/api/deployments/${currentDeploy.id}`);
    currentDeploy = null;
    updateDeployUI();
    await loadAgents();
    toast('Agent undeployed');
  } catch (e) {
    toast('Undeploy failed: ' + e.message, true);
  } finally {
    btn.disabled = false;
  }
});

// ── New agent ─────────────────────────────────────────────────────
document.getElementById('btn-new-agent').addEventListener('click', async () => {
  const defaultConfig = {
    name:               'NewAgent',
    description:        'A new adaptive agent',
    archetype:          'explorer',
    behavior_profile:   { curiosity:0.5, caution:0.5, creativity:0.5, precision:0.5, autonomy:0.5, empathy:0.5 },
    capabilities:       [],
    system_prompt:      '',
    memory_enabled:     true,
    can_spawn_children: false,
    max_children:       0,
    ecosystem_tags:     [],
  };
  try {
    const agent = await api('POST', '/api/agents', defaultConfig);
    agents.push(agent);
    currentAgent = agent;
    renderSidebar();
    renderAgentPanel(agent);
    // Switch to Design tab
    activateTab('design');
    toast('New agent created — configure in the DESIGN tab');
  } catch (e) {
    toast('Create failed: ' + e.message, true);
  }
});

// ── Refresh & Delete ──────────────────────────────────────────────
document.getElementById('btn-refresh-agent').addEventListener('click', async () => {
  if (!currentAgent) return;
  try {
    const fresh = await api('GET', `/api/agents/${currentAgent.id}`);
    currentAgent = fresh;
    agents = agents.map(a => a.id === fresh.id ? fresh : a);
    renderSidebar();
    renderAgentPanel(fresh);
    toast('Refreshed');
  } catch (e) {
    toast('Refresh failed: ' + e.message, true);
  }
});

document.getElementById('btn-delete-agent').addEventListener('click', async () => {
  if (!currentAgent) return;
  if (!confirm(`Delete agent "${currentAgent.config.name}"?`)) return;
  try {
    await api('DELETE', `/api/agents/${currentAgent.id}`);
    agents = agents.filter(a => a.id !== currentAgent.id);
    currentAgent = null;
    renderSidebar();
    document.getElementById('agent-panel').classList.add('hidden');
    document.getElementById('main-empty').classList.remove('hidden');
    toast('Agent deleted');
  } catch (e) {
    toast('Delete failed: ' + e.message, true);
  }
});

// ── Tab switching ─────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    activateTab(tab.dataset.tab);
    if (tab.dataset.tab === 'memory') loadMemory();
    if (tab.dataset.tab === 'deploy' && currentAgent) renderDeployTab(currentAgent);
  });
});

function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === name)
  );
  document.querySelectorAll('.tab-content').forEach(c =>
    c.classList.toggle('active', c.id === `tab-${name}`)
  );
}

// ── Helpers ───────────────────────────────────────────────────────
function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ──────────────────────────────────────────────────────────
(async () => {
  await loadAgents();
  // Poll every 15 s to reflect external changes
  setInterval(loadAgents, 15000);
})();
