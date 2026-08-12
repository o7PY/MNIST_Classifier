async function loadPresets() {
  const container = document.getElementById("presets-list");
  try {
    const res = await fetch("/api/presets");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const presetList = await res.json();

    if (presetList.length === 0) {
      container.innerHTML = '<p class="placeholder-note">No presets available.</p>';
      return;
    }

    container.innerHTML = "";
    for (const preset of presetList) {
      const card = document.createElement("div");
      card.className = "preset-card";

      const summary = preset.nodes.map((n) => n.type).join(" → ");

      const title = document.createElement("h3");
      title.textContent = preset.name;

      const desc = document.createElement("p");
      desc.textContent = preset.description;

      const chain = document.createElement("p");
      chain.className = "preset-summary";
      chain.textContent = summary;

      const meta = document.createElement("p");
      meta.className = "placeholder-note";
      meta.textContent = `${preset.nodes.length} nodes, ${preset.edges.length} edges`;

      const loadBtn = document.createElement("button");
      loadBtn.type = "button";
      loadBtn.className = "palette-btn";
      loadBtn.textContent = "Load into Editor";
      loadBtn.addEventListener("click", () => {
        if (window.graphEditor) window.graphEditor.loadGraph(preset);
        const nameInput = document.getElementById("model-name-input");
        if (nameInput) nameInput.value = preset.name;
        setSaveStatus("", "");
      });

      card.append(title, desc, chain, meta, loadBtn);
      container.appendChild(card);
    }
  } catch (err) {
    container.innerHTML = `<p class="placeholder-note">Failed to load presets: ${err.message}</p>`;
  }
}

async function loadModels() {
  const container = document.getElementById("models-list");
  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const models = await res.json();

    if (models.length === 0) {
      container.innerHTML = '<p class="placeholder-note">No saved models yet.</p>';
    } else {
      container.innerHTML = "";
      for (const model of models) {
        const row = document.createElement("div");
        row.className = "model-row";

        const label = document.createElement("span");
        label.className = "model-row-label";
        let text = `${model.name} — ${model.status}`;
        if (model.status === "trained" && typeof model.val_f1 === "number") {
          text += ` (val_f1=${model.val_f1.toFixed(4)}, val_acc=${model.val_acc.toFixed(4)})`;
        }
        label.textContent = text;
        row.appendChild(label);

        const actions = document.createElement("div");
        actions.className = "model-row-actions";

        if (model.status === "untrained" || model.status === "failed") {
          const trainBtn = document.createElement("button");
          trainBtn.type = "button";
          trainBtn.className = "palette-btn";
          trainBtn.textContent = "Train";
          trainBtn.addEventListener("click", () => trainModel(model.model_id));
          actions.appendChild(trainBtn);
        }

        const renameBtn = document.createElement("button");
        renameBtn.type = "button";
        renameBtn.className = "palette-btn";
        renameBtn.textContent = "Rename";
        renameBtn.addEventListener("click", () => startRename(model, label));
        actions.appendChild(renameBtn);

        actions.appendChild(makeDeleteButton(model));

        row.appendChild(actions);
        container.appendChild(row);
      }
    }

    const trainingModel = models.find((m) => m.status === "training");
    if (trainingModel && (!window.currentTrainSocket || window.currentTrainSocket.readyState !== WebSocket.OPEN)) {
      startTrainSocket(trainingModel.model_id);
    }
  } catch (err) {
    container.innerHTML = `<p class="placeholder-note">Failed to load saved models: ${err.message}</p>`;
  }
}

function setTrainError(text) {
  const el = document.getElementById("train-error-line");
  el.textContent = text;
  el.className = text ? "save-status save-status-bad" : "save-status";
}

function startRename(model, labelEl) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "rename-input";
  input.value = model.name;

  let done = false;
  const save = async () => {
    if (done) return;
    done = true;
    const newName = input.value.trim();
    if (newName && newName !== model.name) {
      await fetch(`/api/models/${encodeURIComponent(model.model_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
      });
    }
    loadModels();
  };

  input.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") save();
    if (ev.key === "Escape") {
      done = true;
      loadModels();
    }
  });
  input.addEventListener("blur", save);

  labelEl.replaceWith(input);
  input.focus();
  input.select();
}

function makeDeleteButton(model) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "palette-btn palette-btn-danger";
  btn.textContent = "Delete";

  let confirming = false;
  let resetTimer = null;

  btn.addEventListener("click", async () => {
    if (!confirming) {
      confirming = true;
      btn.textContent = "Confirm delete?";
      resetTimer = setTimeout(() => {
        confirming = false;
        btn.textContent = "Delete";
      }, 3000);
      return;
    }
    clearTimeout(resetTimer);
    setTrainError("");
    const res = await fetch(`/api/models/${encodeURIComponent(model.model_id)}`, { method: "DELETE" });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      setTrainError(`Could not delete "${model.name}": ${data.detail || res.status}`);
      confirming = false;
      btn.textContent = "Delete";
      return;
    }
    loadModels();
  });

  return btn;
}

function startTrainSocket(jobId) {
  if (window.currentTrainSocket) window.currentTrainSocket.close();
  window.currentTrainSocket = window.connectTrainingSocket(jobId, { onMessage: updateProgressUI });
}

function updateProgressUI(data) {
  const card = document.getElementById("training-progress-card");
  const statusLine = document.getElementById("training-status-line");
  const fill = document.getElementById("training-progress-fill");
  const metricsLine = document.getElementById("training-metrics-line");
  card.style.display = "";

  if (data.status === "unknown_job") {
    statusLine.textContent = "No active training job for this session.";
    fill.style.width = "0%";
    metricsLine.textContent = "";
    return;
  }

  if (data.status === "done") {
    statusLine.textContent = `Training complete for "${data.model_id}". Best val F1: ${(data.best_val_f1 ?? 0).toFixed(4)}, best val acc: ${(data.best_val_acc ?? 0).toFixed(4)}`;
    fill.style.width = "100%";
    loadModels();
    return;
  }

  if (data.status === "failed") {
    statusLine.textContent = `Training failed for "${data.model_id}": ${data.error}`;
    fill.style.width = "0%";
    loadModels();
    return;
  }

  const maxEpochs = Number(document.getElementById("hp-epochs")?.value) || 20;
  const latest = data.latest || (data.history && data.history.length ? data.history[data.history.length - 1] : null);
  if (latest) {
    statusLine.textContent = `Model "${data.model_id}" — epoch ${latest.epoch}/${maxEpochs}`;
    fill.style.width = `${Math.min(100, (latest.epoch / maxEpochs) * 100)}%`;
    metricsLine.textContent =
      `train_loss=${latest.train_loss.toFixed(4)} train_acc=${latest.train_acc.toFixed(4)} | ` +
      `val_loss=${latest.val_loss.toFixed(4)} val_acc=${latest.val_acc.toFixed(4)} val_f1=${latest.val_f1.toFixed(4)}`;
  } else {
    statusLine.textContent = `Model "${data.model_id}" — training started…`;
  }
}

async function trainModel(modelId) {
  setTrainError("");
  const hyperparams = {
    learning_rate: parseFloat(document.getElementById("hp-lr").value),
    batch_size: parseInt(document.getElementById("hp-batch").value, 10),
    max_epochs: parseInt(document.getElementById("hp-epochs").value, 10),
    optimizer: document.getElementById("hp-optimizer").value,
    early_stopping_patience: parseInt(document.getElementById("hp-patience").value, 10),
  };
  try {
    const res = await fetch(`/api/models/${encodeURIComponent(modelId)}/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(hyperparams),
    });
    const data = await res.json();
    if (!res.ok) {
      setTrainError(`Could not start training: ${data.detail}`);
      return;
    }
    startTrainSocket(data.job_id);
    loadModels();
  } catch (err) {
    setTrainError(`Training request failed: ${err.message}`);
  }
}

function setSaveStatus(text, kind) {
  const statusEl = document.getElementById("save-status");
  statusEl.textContent = text;
  statusEl.className = kind ? `save-status save-status-${kind}` : "save-status";
}

async function saveModel() {
  const nameInput = document.getElementById("model-name-input");
  const name = nameInput.value.trim();
  window.graphEditor.clearErrorHighlight();

  if (!name) {
    setSaveStatus("Enter a model name first.", "bad");
    return;
  }

  const graphData = window.graphEditor.getGraph();
  setSaveStatus("Validating & saving…", "");

  try {
    const res = await fetch("/api/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, nodes: graphData.nodes, edges: graphData.edges, source: "custom" }),
    });
    const data = await res.json();

    if (!res.ok) {
      const detail = data.detail || {};
      setSaveStatus(`Validation failed: ${detail.message || JSON.stringify(data)}`, "bad");
      if (detail.node_id) window.graphEditor.highlightError(detail.node_id);
      return;
    }

    setSaveStatus(`Saved as "${name}" (${data.model_id}).`, "good");
    nameInput.value = "";
    loadModels();
  } catch (err) {
    setSaveStatus(`Request failed: ${err.message}`, "bad");
  }
}

window.graphEditor = new GraphEditor(document.getElementById("graph-editor-root"));
document.getElementById("save-model-btn").addEventListener("click", saveModel);

loadPresets();
loadModels();
