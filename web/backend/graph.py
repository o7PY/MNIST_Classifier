"""Authoritative DAG validation and shape inference for architecture graphs.

This is the source of truth referenced by webspec.md's "Server-side
validation" section. frontend/js/graph_editor.js has its own best-effort
mirror for live UX feedback, but this module is what POST /api/models
actually enforces.
"""

from __future__ import annotations

from dataclasses import dataclass


class GraphValidationError(Exception):
    def __init__(self, message: str, node_id: str | None = None):
        self.message = message
        self.node_id = node_id
        super().__init__(message)


@dataclass
class Shape:
    kind: str  # "spatial" or "flat"
    c: int = 0
    h: int = 0
    w: int = 0
    n: int = 0

    def __eq__(self, other):
        if not isinstance(other, Shape) or self.kind != other.kind:
            return False
        if self.kind == "spatial":
            return (self.c, self.h, self.w) == (other.c, other.h, other.w)
        return self.n == other.n

    def label(self) -> str:
        if self.kind == "spatial":
            return f"{self.c}x{self.h}x{self.w}"
        return str(self.n)


MERGE_TYPES = {"Add", "Concat"}
ALL_TYPES = {
    "Input", "Output", "Conv2d", "MaxPool2d", "Flatten", "Linear",
    "Activation", "Dropout", "BatchNorm", "Add", "Concat",
}


def validate_and_infer(nodes: list[dict], edges: list[dict]) -> dict[str, Shape]:
    """Validate a graph per webspec.md's rules; return {node_id: Shape} on success.

    Raises GraphValidationError with a specific, node-indexed message on the
    first violation found. `nodes`/`edges` are plain dicts:
    node = {"id": str, "type": str, "config": dict}, edge = {"from": str, "to": str}.
    """
    if not nodes:
        raise GraphValidationError("Graph is empty: no nodes.")

    node_map: dict[str, dict] = {}
    for n in nodes:
        if n["id"] in node_map:
            raise GraphValidationError(f"Duplicate node id '{n['id']}'.", n["id"])
        node_map[n["id"]] = n

    for n in nodes:
        if n["type"] not in ALL_TYPES:
            raise GraphValidationError(f"Node '{n['id']}': unknown block type '{n['type']}'.", n["id"])

    input_nodes = [n for n in nodes if n["type"] == "Input"]
    output_nodes = [n for n in nodes if n["type"] == "Output"]
    if len(input_nodes) != 1:
        raise GraphValidationError(f"Graph must have exactly one Input node (found {len(input_nodes)}).")
    if len(output_nodes) != 1:
        raise GraphValidationError(f"Graph must have exactly one Output node (found {len(output_nodes)}).")
    input_id = input_nodes[0]["id"]
    output_id = output_nodes[0]["id"]

    for e in edges:
        if e["from"] not in node_map:
            raise GraphValidationError(f"Edge references unknown node '{e['from']}'.")
        if e["to"] not in node_map:
            raise GraphValidationError(f"Edge references unknown node '{e['to']}'.")
        if e["to"] == input_id:
            raise GraphValidationError(f"Node '{input_id}' (Input) cannot have incoming edges.", input_id)
        if e["from"] == output_id:
            raise GraphValidationError(f"Node '{output_id}' (Output) cannot have outgoing edges.", output_id)

    indeg = {nid: 0 for nid in node_map}
    outdeg = {nid: 0 for nid in node_map}
    adj: dict[str, list[str]] = {nid: [] for nid in node_map}
    preds: dict[str, list[str]] = {nid: [] for nid in node_map}
    for e in edges:
        adj[e["from"]].append(e["to"])
        preds[e["to"]].append(e["from"])
        indeg[e["to"]] += 1
        outdeg[e["from"]] += 1

    for nid, n in node_map.items():
        if n["type"] != "Input" and indeg[nid] == 0:
            raise GraphValidationError(f"Node '{nid}' ({n['type']}) is disconnected: no incoming edges.", nid)
        if n["type"] != "Output" and outdeg[nid] == 0:
            raise GraphValidationError(f"Node '{nid}' ({n['type']}) is a dead end: no outgoing edges.", nid)

    for nid, n in node_map.items():
        if n["type"] in MERGE_TYPES and indeg[nid] < 2:
            raise GraphValidationError(
                f"Node '{nid}' ({n['type']}) needs at least 2 incoming edges (found {indeg[nid]}).", nid
            )

    local_indeg = dict(indeg)
    queue = [nid for nid in node_map if local_indeg[nid] == 0]
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for nxt in adj[nid]:
            local_indeg[nxt] -= 1
            if local_indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(node_map):
        raise GraphValidationError("Graph contains a cycle (must be a DAG).")

    shapes: dict[str, Shape] = {input_id: Shape(kind="spatial", c=1, h=28, w=28)}
    for nid in order:
        if nid == input_id:
            continue
        node = node_map[nid]
        pred_shapes = [shapes[p] for p in preds[nid]]
        shapes[nid] = _infer_node_shape(node, pred_shapes)
    return shapes


def _infer_node_shape(node: dict, pred_shapes: list[Shape]) -> Shape:
    nid = node["id"]
    ntype = node["type"]
    cfg = node.get("config") or {}
    s = pred_shapes[0]

    if ntype == "Conv2d":
        if s.kind != "spatial":
            raise GraphValidationError(f"Node '{nid}' (Conv2d): expects a spatial input, got a flat tensor.", nid)
        k = _positive_int(cfg.get("kernel_size"), nid, "kernel_size")
        st = _positive_int(cfg.get("stride", 1), nid, "stride")
        p = _nonneg_int(cfg.get("padding", 0), nid, "padding")
        oc = _positive_int(cfg.get("out_channels"), nid, "out_channels")
        h = (s.h + 2 * p - k) // st + 1
        w = (s.w + 2 * p - k) // st + 1
        if h <= 0 or w <= 0:
            raise GraphValidationError(
                f"Node '{nid}' (Conv2d): input spatial size {s.h}x{s.w} too small for kernel_size={k}.", nid
            )
        return Shape(kind="spatial", c=oc, h=h, w=w)

    if ntype == "MaxPool2d":
        if s.kind != "spatial":
            raise GraphValidationError(f"Node '{nid}' (MaxPool2d): expects a spatial input, got a flat tensor.", nid)
        k = _positive_int(cfg.get("kernel_size"), nid, "kernel_size")
        h = s.h // k
        w = s.w // k
        if h <= 0 or w <= 0:
            raise GraphValidationError(
                f"Node '{nid}' (MaxPool2d): input spatial size {s.h}x{s.w} too small for kernel_size={k}.", nid
            )
        return Shape(kind="spatial", c=s.c, h=h, w=w)

    if ntype == "Flatten":
        if s.kind != "spatial":
            raise GraphValidationError(f"Node '{nid}' (Flatten): input is already flat.", nid)
        return Shape(kind="flat", n=s.c * s.h * s.w)

    if ntype == "Linear":
        if s.kind != "flat":
            raise GraphValidationError(
                f"Node '{nid}' (Linear): expects a flat input (add a Flatten first), got a spatial tensor.", nid
            )
        n = _positive_int(cfg.get("out_features"), nid, "out_features")
        return Shape(kind="flat", n=n)

    if ntype == "Dropout":
        if s.kind != "flat":
            raise GraphValidationError(f"Node '{nid}' (Dropout): expects a flat input, got a spatial tensor.", nid)
        p = cfg.get("p", 0.0)
        if not isinstance(p, (int, float)) or isinstance(p, bool) or not (0 <= p <= 0.9):
            raise GraphValidationError(f"Node '{nid}' (Dropout): p must be in [0, 0.9], got {p!r}.", nid)
        return s

    if ntype == "BatchNorm":
        return s

    if ntype == "Activation":
        allowed = {"ReLU", "GELU", "Sigmoid"}
        t = cfg.get("type")
        if t not in allowed:
            raise GraphValidationError(
                f"Node '{nid}' (Activation): type must be one of {sorted(allowed)}, got {t!r}.", nid
            )
        return s

    if ntype == "Add":
        first = pred_shapes[0]
        for p_shape in pred_shapes[1:]:
            if p_shape != first:
                raise GraphValidationError(
                    f"Node '{nid}' (Add): input shapes {first.label()} and {p_shape.label()} don't match.", nid
                )
        return first

    if ntype == "Concat":
        kind = pred_shapes[0].kind
        if any(p.kind != kind for p in pred_shapes):
            raise GraphValidationError(f"Node '{nid}' (Concat): cannot mix spatial and flat inputs.", nid)
        if kind == "spatial":
            h, w = pred_shapes[0].h, pred_shapes[0].w
            if any((p.h, p.w) != (h, w) for p in pred_shapes):
                raise GraphValidationError(
                    f"Node '{nid}' (Concat): input spatial dims must match (got differing H/W).", nid
                )
            c = sum(p.c for p in pred_shapes)
            return Shape(kind="spatial", c=c, h=h, w=w)
        n = sum(p.n for p in pred_shapes)
        return Shape(kind="flat", n=n)

    if ntype == "Output":
        if s.kind != "flat":
            raise GraphValidationError(
                f"Node '{nid}' (Output): expects a flat input (add a Flatten first), got a spatial tensor.", nid
            )
        return Shape(kind="flat", n=10)

    raise GraphValidationError(f"Node '{nid}': unhandled block type '{ntype}'.", nid)


def _positive_int(value, nid, field):
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise GraphValidationError(f"Node '{nid}': '{field}' must be a positive integer, got {value!r}.", nid)
    if v <= 0:
        raise GraphValidationError(f"Node '{nid}': '{field}' must be a positive integer, got {v}.", nid)
    return v


def _nonneg_int(value, nid, field):
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise GraphValidationError(f"Node '{nid}': '{field}' must be a non-negative integer, got {value!r}.", nid)
    if v < 0:
        raise GraphValidationError(f"Node '{nid}': '{field}' must be a non-negative integer, got {v}.", nid)
    return v
