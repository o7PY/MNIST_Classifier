"""Compiles a validated node/edge graph into a PyTorch nn.Module.

forward() is a small graph interpreter: it evaluates nodes in topological
order, caching each node's output tensor for downstream consumers. Add/Concat
are pure ops (no learnable parameters); every other block type maps to one
nn.Module, sized using the shapes graph.py already computed during
validation.
"""

import torch
import torch.nn as nn

from . import graph as graph_module

_ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "Sigmoid": nn.Sigmoid}


def _topo_order(nodes: list[dict], edges: list[dict]):
    node_ids = [n["id"] for n in nodes]
    adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
    preds: dict[str, list[str]] = {nid: [] for nid in node_ids}
    indeg = {nid: 0 for nid in node_ids}
    for e in edges:
        adj[e["from"]].append(e["to"])
        preds[e["to"]].append(e["from"])
        indeg[e["to"]] += 1

    queue = [nid for nid in node_ids if indeg[nid] == 0]
    order: list[str] = []
    local_indeg = dict(indeg)
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for nxt in adj[nid]:
            local_indeg[nxt] -= 1
            if local_indeg[nxt] == 0:
                queue.append(nxt)
    return order, preds


def _make_layer(node: dict, input_shape: graph_module.Shape):
    ntype = node["type"]
    cfg = node.get("config") or {}

    if ntype == "Conv2d":
        return nn.Conv2d(
            in_channels=input_shape.c,
            out_channels=int(cfg["out_channels"]),
            kernel_size=int(cfg["kernel_size"]),
            stride=int(cfg.get("stride", 1)),
            padding=int(cfg.get("padding", 0)),
        )
    if ntype == "MaxPool2d":
        return nn.MaxPool2d(kernel_size=int(cfg["kernel_size"]))
    if ntype == "Flatten":
        return nn.Flatten()
    if ntype == "Linear":
        return nn.Linear(input_shape.n, int(cfg["out_features"]))
    if ntype == "Activation":
        return _ACTIVATIONS[cfg.get("type", "ReLU")]()
    if ntype == "Dropout":
        return nn.Dropout(p=float(cfg.get("p", 0.0)))
    if ntype == "BatchNorm":
        if input_shape.kind == "spatial":
            return nn.BatchNorm2d(input_shape.c)
        return nn.BatchNorm1d(input_shape.n)
    if ntype == "Output":
        return nn.Linear(input_shape.n, 10)
    return None  # Input, Add, Concat: no learnable layer


class GraphModel(nn.Module):
    def __init__(self, nodes: list[dict], edges: list[dict]):
        super().__init__()
        # Re-validate: this is the authoritative check, and it gives us the
        # per-node output shapes needed to size every layer.
        shapes = graph_module.validate_and_infer(nodes, edges)

        node_map = {n["id"]: n for n in nodes}
        order, preds = _topo_order(nodes, edges)
        input_id = next(n["id"] for n in nodes if n["type"] == "Input")
        output_id = next(n["id"] for n in nodes if n["type"] == "Output")

        layers = {}
        for nid in order:
            if nid == input_id:
                continue
            node = node_map[nid]
            if node["type"] in ("Add", "Concat"):
                continue
            pred_shape = shapes[preds[nid][0]]
            layer = _make_layer(node, pred_shape)
            if layer is not None:
                layers[nid] = layer

        self.layers = nn.ModuleDict(layers)
        self._order = order
        self._preds = preds
        self._node_map = node_map
        self._input_id = input_id
        self._output_id = output_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = {self._input_id: x}
        for nid in self._order:
            if nid == self._input_id:
                continue
            node = self._node_map[nid]
            pred_tensors = [outputs[p] for p in self._preds[nid]]

            if node["type"] == "Add":
                out = pred_tensors[0]
                for t in pred_tensors[1:]:
                    out = out + t
            elif node["type"] == "Concat":
                out = torch.cat(pred_tensors, dim=1)
            else:
                out = self.layers[nid](pred_tensors[0])

            outputs[nid] = out
        return outputs[self._output_id]
