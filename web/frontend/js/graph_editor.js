/*
 * Client-side node-graph editor. This is intentionally backend-validation
 * -free (Phase 3): shape inference here is a best-effort live preview for
 * the user, not authoritative. The real, authoritative DAG validation and
 * shape inference lives server-side in Phase 4's graph.py.
 */

const BLOCK_TYPES = [
  "Conv2d",
  "MaxPool2d",
  "Flatten",
  "Linear",
  "Activation",
  "Dropout",
  "BatchNorm",
  "Add",
  "Concat",
];

function defaultConfig(type) {
  switch (type) {
    case "Conv2d":
      return { out_channels: 16, kernel_size: 3, stride: 1, padding: 0 };
    case "MaxPool2d":
      return { kernel_size: 2 };
    case "Linear":
      return { out_features: 128 };
    case "Activation":
      return { type: "ReLU" };
    case "Dropout":
      return { p: 0.3 };
    default:
      return {};
  }
}

function toNum(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function shapesEqual(a, b) {
  if (!a || !b || a.kind !== b.kind) return false;
  if (a.kind === "spatial") return a.c === b.c && a.h === b.h && a.w === b.w;
  if (a.kind === "flat") return a.n === b.n;
  return false;
}

function computeNodeShape(node, predShapes) {
  const cfg = node.config || {};
  const s = predShapes[0];

  switch (node.type) {
    case "Conv2d": {
      if (!s || s.kind !== "spatial") return null;
      const k = toNum(cfg.kernel_size);
      const st = toNum(cfg.stride) || 1;
      const p = toNum(cfg.padding) ?? 0;
      const oc = toNum(cfg.out_channels);
      if (!k || k <= 0 || !oc || oc <= 0 || st <= 0) return null;
      const h = Math.floor((s.h + 2 * p - k) / st) + 1;
      const w = Math.floor((s.w + 2 * p - k) / st) + 1;
      if (h <= 0 || w <= 0) return { kind: "invalid" };
      return { kind: "spatial", c: oc, h, w };
    }
    case "MaxPool2d": {
      if (!s || s.kind !== "spatial") return null;
      const k = toNum(cfg.kernel_size);
      if (!k || k <= 0) return null;
      const h = Math.floor(s.h / k);
      const w = Math.floor(s.w / k);
      if (h <= 0 || w <= 0) return { kind: "invalid" };
      return { kind: "spatial", c: s.c, h, w };
    }
    case "Flatten": {
      if (!s) return null;
      if (s.kind === "spatial") return { kind: "flat", n: s.c * s.h * s.w };
      return s;
    }
    case "Linear": {
      const n = toNum(cfg.out_features);
      if (!n || n <= 0) return null;
      if (!s || s.kind !== "flat") return { kind: "invalid" };
      return { kind: "flat", n };
    }
    case "Dropout":
    case "BatchNorm":
    case "Activation":
      return s || null;
    case "Add": {
      if (predShapes.length < 2) return null;
      const first = predShapes[0];
      const allMatch = predShapes.every((p) => shapesEqual(p, first));
      return allMatch ? first : { kind: "mismatch" };
    }
    case "Concat": {
      if (predShapes.length < 2) return null;
      const kind = predShapes[0].kind;
      if (!predShapes.every((p) => p.kind === kind)) return { kind: "mismatch" };
      if (kind === "spatial") {
        const h = predShapes[0].h;
        const w = predShapes[0].w;
        if (!predShapes.every((p) => p.h === h && p.w === w)) return { kind: "mismatch" };
        const c = predShapes.reduce((a, p) => a + p.c, 0);
        return { kind: "spatial", c, h, w };
      }
      const n = predShapes.reduce((a, p) => a + p.n, 0);
      return { kind: "flat", n };
    }
    case "Output":
      return { kind: "flat", n: 10 };
    default:
      return null;
  }
}

function shapeLabel(shape) {
  if (!shape) return "–";
  if (shape.kind === "spatial") return `${shape.c}×${shape.h}×${shape.w}`;
  if (shape.kind === "flat") return `${shape.n}`;
  if (shape.kind === "invalid") return "invalid";
  if (shape.kind === "mismatch") return "mismatch";
  return "?";
}

class GraphEditor {
  constructor(rootEl) {
    this.root = rootEl;
    this.nodeCounters = {};
    this.dragNodeId = null;
    this._buildChrome();
    this._resetGraph();
    this._render();
  }

  // ---- public API -------------------------------------------------------

  addNode(type, x, y) {
    const id = this._generateId(type);
    this.nodes[id] = { id, type, config: defaultConfig(type), x, y };
    this._render();
    return id;
  }

  removeNode(id) {
    if (id === "input" || id === "output") return;
    delete this.nodes[id];
    this.edges = this.edges.filter((e) => e.from !== id && e.to !== id);
    this._render();
  }

  addEdge(from, to) {
    if (from === to) return;
    if (!this.nodes[from] || !this.nodes[to]) return;
    if (to === "input" || from === "output") return;
    if (this.edges.some((e) => e.from === from && e.to === to)) return;
    this.edges.push({ from, to });
    this._render();
  }

  removeEdge(index) {
    this.edges.splice(index, 1);
    this._render();
  }

  clear() {
    this._resetGraph();
    this._render();
  }

  loadGraph(graph) {
    this.nodes = {};
    for (const n of graph.nodes) {
      this.nodes[n.id] = { id: n.id, type: n.type, config: { ...n.config }, x: 0, y: 0 };
    }
    this.edges = graph.edges.map((e) => ({ from: e.from, to: e.to }));
    this._autoLayout();
    this._render();
  }

  getGraph() {
    return {
      nodes: Object.values(this.nodes).map((n) => ({ id: n.id, type: n.type, config: n.config })),
      edges: this.edges.map((e) => ({ from: e.from, to: e.to })),
    };
  }

  highlightError(nodeId) {
    this.clearErrorHighlight();
    const el = this.nodeLayer.querySelector(`.graph-node[data-id="${cssEscape(nodeId)}"]`);
    if (el) el.classList.add("node-error");
  }

  clearErrorHighlight() {
    for (const el of this.nodeLayer.querySelectorAll(".node-error")) el.classList.remove("node-error");
  }

  // ---- internal: state ----------------------------------------------------

  _resetGraph() {
    this.nodes = {
      input: { id: "input", type: "Input", config: {}, x: 40, y: 180 },
      output: { id: "output", type: "Output", config: {}, x: 760, y: 180 },
    };
    this.edges = [];
    this.nodeCounters = {};
  }

  _generateId(type) {
    const key = type.toLowerCase();
    this.nodeCounters[key] = (this.nodeCounters[key] || 0) + 1;
    let id = `${key}_${this.nodeCounters[key]}`;
    while (this.nodes[id]) {
      this.nodeCounters[key] += 1;
      id = `${key}_${this.nodeCounters[key]}`;
    }
    return id;
  }

  _autoLayout() {
    const ids = Object.keys(this.nodes);
    const indeg = {};
    const adj = {};
    for (const id of ids) {
      indeg[id] = 0;
      adj[id] = [];
    }
    for (const e of this.edges) {
      if (!(e.from in this.nodes) || !(e.to in this.nodes)) continue;
      adj[e.from].push(e.to);
      indeg[e.to] += 1;
    }
    const depth = {};
    const localIndeg = { ...indeg };
    const queue = ids.filter((id) => indeg[id] === 0);
    for (const id of queue) depth[id] = 0;
    while (queue.length) {
      const id = queue.shift();
      for (const nxt of adj[id]) {
        depth[nxt] = Math.max(depth[nxt] ?? 0, (depth[id] ?? 0) + 1);
        localIndeg[nxt] -= 1;
        if (localIndeg[nxt] === 0) queue.push(nxt);
      }
    }
    const byDepth = {};
    for (const id of ids) {
      const d = depth[id] ?? 0;
      (byDepth[d] = byDepth[d] || []).push(id);
    }
    const xGap = 200;
    const yGap = 140;
    for (const d of Object.keys(byDepth)) {
      byDepth[d].forEach((id, i) => {
        this.nodes[id].x = 40 + Number(d) * xGap;
        this.nodes[id].y = 40 + i * yGap;
      });
    }
  }

  _computeShapes() {
    const ids = Object.keys(this.nodes);
    const indeg = {};
    const adj = {};
    for (const id of ids) {
      indeg[id] = 0;
      adj[id] = [];
    }
    for (const e of this.edges) {
      if (!(e.from in this.nodes) || !(e.to in this.nodes)) continue;
      adj[e.from].push(e.to);
      indeg[e.to] += 1;
    }
    const shapes = {};
    if (this.nodes.input) shapes.input = { kind: "spatial", c: 1, h: 28, w: 28 };

    const localIndeg = { ...indeg };
    const queue = ids.filter((id) => indeg[id] === 0);
    const order = [];
    while (queue.length) {
      const id = queue.shift();
      order.push(id);
      for (const nxt of adj[id]) {
        localIndeg[nxt] -= 1;
        if (localIndeg[nxt] === 0) queue.push(nxt);
      }
    }

    for (const id of order) {
      if (id === "input") continue;
      const preds = this.edges.filter((e) => e.to === id && e.from in this.nodes).map((e) => e.from);
      if (preds.length === 0) {
        shapes[id] = null;
        continue;
      }
      const predShapes = preds.map((p) => shapes[p]);
      if (predShapes.some((s) => s === undefined || s === null)) {
        shapes[id] = null;
        continue;
      }
      shapes[id] = computeNodeShape(this.nodes[id], predShapes);
    }
    return shapes;
  }

  // ---- internal: DOM ------------------------------------------------------

  _buildChrome() {
    this.root.innerHTML = "";

    this.paletteEl = document.createElement("div");
    this.paletteEl.className = "graph-palette";
    for (const type of BLOCK_TYPES) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "palette-btn";
      btn.textContent = type;
      btn.addEventListener("click", () => {
        const n = Object.keys(this.nodes).length;
        this.addNode(type, 260 + ((n * 24) % 200), 40 + ((n * 37) % 260));
      });
      this.paletteEl.appendChild(btn);
    }
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "palette-btn palette-btn-danger";
    clearBtn.textContent = "Clear canvas";
    clearBtn.addEventListener("click", () => this.clear());
    this.paletteEl.appendChild(clearBtn);

    this.canvasEl = document.createElement("div");
    this.canvasEl.className = "graph-canvas";

    this.svgEl = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    this.svgEl.setAttribute("class", "graph-edges-svg");
    this.tempPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
    this.tempPath.setAttribute("class", "edge-temp");
    this.tempPath.style.display = "none";
    this.svgEl.appendChild(this.tempPath);

    this.nodeLayer = document.createElement("div");
    this.nodeLayer.className = "graph-node-layer";

    this.canvasEl.append(this.svgEl, this.nodeLayer);

    this.jsonDetails = document.createElement("details");
    this.jsonDetails.className = "graph-json-details";
    const summary = document.createElement("summary");
    summary.textContent = "Graph JSON (debug)";
    this.jsonOutput = document.createElement("pre");
    this.jsonOutput.id = "graph-json-output";
    this.jsonDetails.append(summary, this.jsonOutput);

    this.root.append(this.paletteEl, this.canvasEl, this.jsonDetails);
  }

  _portCenter(nodeId, which) {
    const nodeEl = this.nodeLayer.querySelector(`.graph-node[data-id="${cssEscape(nodeId)}"]`);
    if (!nodeEl) return null;
    const portEl = nodeEl.querySelector(which === "out" ? ".port-out" : ".port-in");
    if (!portEl) return null;
    const portRect = portEl.getBoundingClientRect();
    const canvasRect = this.canvasEl.getBoundingClientRect();
    return {
      x: portRect.left + portRect.width / 2 - canvasRect.left + this.canvasEl.scrollLeft,
      y: portRect.top + portRect.height / 2 - canvasRect.top + this.canvasEl.scrollTop,
    };
  }

  _edgePathD(p1, p2) {
    const dx = Math.max(40, Math.abs(p2.x - p1.x) / 2);
    return `M ${p1.x} ${p1.y} C ${p1.x + dx} ${p1.y}, ${p2.x - dx} ${p2.y}, ${p2.x} ${p2.y}`;
  }

  _refreshEdgePositions() {
    for (const g of this.svgEl.querySelectorAll(".edge")) {
      const from = g.dataset.from;
      const to = g.dataset.to;
      const p1 = this._portCenter(from, "out");
      const p2 = this._portCenter(to, "in");
      if (!p1 || !p2) continue;
      const d = this._edgePathD(p1, p2);
      for (const path of g.querySelectorAll("path")) path.setAttribute("d", d);
    }
  }

  _updateShapeLabels() {
    const shapes = this._computeShapes();
    for (const id in this.nodes) {
      const nodeEl = this.nodeLayer.querySelector(`.graph-node[data-id="${cssEscape(id)}"]`);
      if (!nodeEl) continue;
      const shapeEl = nodeEl.querySelector(".node-shape");
      if (!shapeEl) continue;
      const label = shapeLabel(shapes[id]);
      shapeEl.textContent = `shape: ${label}`;
      shapeEl.classList.toggle("shape-bad", label === "invalid" || label === "mismatch");
    }
    this.jsonOutput.textContent = JSON.stringify(this.getGraph(), null, 2);
  }

  _makePort(kind) {
    const el = document.createElement("div");
    el.className = `port port-${kind}`;
    el.title = kind === "out" ? "Drag to connect" : "Drop connection here";
    return el;
  }

  _buildConfigFields(node) {
    const wrap = document.createElement("div");
    wrap.className = "node-config";

    const addField = (label, field, inputEl) => {
      const row = document.createElement("label");
      row.className = "node-config-row";
      const span = document.createElement("span");
      span.textContent = label;
      row.append(span, inputEl);
      wrap.appendChild(row);
    };

    const numberInput = (field, value, opts = {}) => {
      const input = document.createElement("input");
      input.type = "number";
      input.dataset.field = field;
      input.value = value;
      if (opts.min !== undefined) input.min = opts.min;
      if (opts.max !== undefined) input.max = opts.max;
      if (opts.step !== undefined) input.step = opts.step;
      input.addEventListener("mousedown", (ev) => ev.stopPropagation());
      input.addEventListener("input", () => {
        const v = parseFloat(input.value);
        node.config[field] = Number.isFinite(v) ? v : input.value;
        this._updateShapeLabels();
      });
      return input;
    };

    switch (node.type) {
      case "Conv2d":
        addField("out_channels", "out_channels", numberInput("out_channels", node.config.out_channels, { min: 1, step: 1 }));
        addField("kernel_size", "kernel_size", numberInput("kernel_size", node.config.kernel_size, { min: 1, step: 1 }));
        addField("stride", "stride", numberInput("stride", node.config.stride, { min: 1, step: 1 }));
        addField("padding", "padding", numberInput("padding", node.config.padding, { min: 0, step: 1 }));
        break;
      case "MaxPool2d":
        addField("kernel_size", "kernel_size", numberInput("kernel_size", node.config.kernel_size, { min: 1, step: 1 }));
        break;
      case "Linear":
        addField("out_features", "out_features", numberInput("out_features", node.config.out_features, { min: 1, step: 1 }));
        break;
      case "Dropout":
        addField("p", "p", numberInput("p", node.config.p, { min: 0, max: 0.9, step: 0.05 }));
        break;
      case "Activation": {
        const select = document.createElement("select");
        select.dataset.field = "type";
        for (const opt of ["ReLU", "GELU", "Sigmoid"]) {
          const optEl = document.createElement("option");
          optEl.value = opt;
          optEl.textContent = opt;
          if (node.config.type === opt) optEl.selected = true;
          select.appendChild(optEl);
        }
        select.addEventListener("mousedown", (ev) => ev.stopPropagation());
        select.addEventListener("change", () => {
          node.config.type = select.value;
          this._updateShapeLabels();
        });
        addField("type", "type", select);
        break;
      }
      default:
        break;
    }
    return wrap;
  }

  _buildNodeEl(node) {
    const el = document.createElement("div");
    el.className = "graph-node";
    el.dataset.id = node.id;
    el.style.left = `${node.x}px`;
    el.style.top = `${node.y}px`;

    const header = document.createElement("div");
    header.className = "node-header";
    const typeEl = document.createElement("span");
    typeEl.className = "node-type";
    typeEl.textContent = node.type;
    header.appendChild(typeEl);

    if (node.type !== "Input" && node.type !== "Output") {
      const nameEl = document.createElement("span");
      nameEl.className = "node-name";
      nameEl.textContent = node.id;
      header.appendChild(nameEl);

      const delBtn = document.createElement("button");
      delBtn.type = "button";
      delBtn.className = "node-delete";
      delBtn.textContent = "×";
      delBtn.title = "Delete node";
      delBtn.addEventListener("mousedown", (ev) => ev.stopPropagation());
      delBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        this.removeNode(node.id);
      });
      header.appendChild(delBtn);
    }

    header.addEventListener("mousedown", (ev) => this._startNodeDrag(ev, node.id, el));
    el.appendChild(header);

    if (node.type === "Input") {
      const info = document.createElement("div");
      info.className = "node-config";
      info.innerHTML = '<span class="placeholder-note">fixed 1×28×28</span>';
      el.appendChild(info);
    } else if (node.type === "Output") {
      const info = document.createElement("div");
      info.className = "node-config";
      info.innerHTML = '<span class="placeholder-note">fixed Linear(&rarr;10)</span>';
      el.appendChild(info);
    } else {
      el.appendChild(this._buildConfigFields(node));
    }

    const shapeEl = document.createElement("div");
    shapeEl.className = "node-shape";
    shapeEl.textContent = "shape: –";
    el.appendChild(shapeEl);

    if (node.type !== "Input") {
      const portIn = this._makePort("in");
      portIn.addEventListener("mousedown", (ev) => ev.stopPropagation());
      el.appendChild(portIn);
    }
    if (node.type !== "Output") {
      const portOut = this._makePort("out");
      portOut.addEventListener("mousedown", (ev) => this._startEdgeDrag(ev, node.id));
      el.appendChild(portOut);
    }

    return el;
  }

  _startNodeDrag(ev, id, el) {
    if (ev.target.closest(".node-delete")) return;
    ev.preventDefault();
    const startX = ev.clientX;
    const startY = ev.clientY;
    const origX = this.nodes[id].x;
    const origY = this.nodes[id].y;
    el.style.zIndex = 10;

    const onMove = (mv) => {
      const dx = mv.clientX - startX;
      const dy = mv.clientY - startY;
      this.nodes[id].x = origX + dx;
      this.nodes[id].y = origY + dy;
      el.style.left = `${this.nodes[id].x}px`;
      el.style.top = `${this.nodes[id].y}px`;
      this._refreshEdgePositions();
    };
    const onUp = () => {
      el.style.zIndex = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  _startEdgeDrag(ev, sourceId) {
    ev.preventDefault();
    ev.stopPropagation();
    const start = this._portCenter(sourceId, "out");
    if (!start) return;
    this.tempPath.style.display = "";
    this.tempPath.setAttribute("d", this._edgePathD(start, start));

    const onMove = (mv) => {
      const canvasRect = this.canvasEl.getBoundingClientRect();
      const p2 = {
        x: mv.clientX - canvasRect.left + this.canvasEl.scrollLeft,
        y: mv.clientY - canvasRect.top + this.canvasEl.scrollTop,
      };
      this.tempPath.setAttribute("d", this._edgePathD(start, p2));
    };
    const onUp = (mv) => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      this.tempPath.style.display = "none";
      const target = document.elementFromPoint(mv.clientX, mv.clientY);
      const targetPort = target && target.closest(".port-in");
      if (targetPort) {
        const targetNode = targetPort.closest(".graph-node");
        if (targetNode) this.addEdge(sourceId, targetNode.dataset.id);
      }
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }

  _render() {
    this.nodeLayer.innerHTML = "";
    for (const id in this.nodes) {
      this.nodeLayer.appendChild(this._buildNodeEl(this.nodes[id]));
    }

    while (this.svgEl.firstChild) this.svgEl.removeChild(this.svgEl.firstChild);
    this.svgEl.appendChild(this.tempPath);

    this.edges.forEach((edge, index) => {
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("class", "edge");
      g.dataset.from = edge.from;
      g.dataset.to = edge.to;

      const hit = document.createElementNS("http://www.w3.org/2000/svg", "path");
      hit.setAttribute("class", "edge-hit");
      const visible = document.createElementNS("http://www.w3.org/2000/svg", "path");
      visible.setAttribute("class", "edge-visible");

      g.append(hit, visible);
      g.addEventListener("click", () => this.removeEdge(index));
      this.svgEl.appendChild(g);
    });

    this._refreshEdgePositions();
    this._updateShapeLabels();
  }
}

function cssEscape(id) {
  return typeof CSS !== "undefined" && CSS.escape ? CSS.escape(id) : id;
}

window.GraphEditor = GraphEditor;
