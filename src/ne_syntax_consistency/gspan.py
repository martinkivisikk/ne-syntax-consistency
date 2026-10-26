"""NE-centred ego graphs and frequent subgraph mining with gSpan.

Each NE span becomes one small labelled graph (its ego graph); gSpan counts
in how many ego graphs a connected subgraph occurs (support).

gSpan works on undirected graphs, so dependency
direction is kept by subdividing every edge: head -h- [deprel] -d- dependent.
A pattern therefore alternates word nodes and relation nodes.
"""

import ast
import collections
import dataclasses
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from gspan_mining import gspan as _gspan
from gspan_mining.graph import AUTO_EDGE_ID, Graph

# Vabamorf case codes in `morph_analysis.form` (e.g. "sg g", "pl in")
CASES = {
    "n",
    "g",
    "p",
    "ill",
    "in",
    "el",
    "all",
    "ad",
    "abl",
    "tr",
    "ter",
    "es",
    "ab",
    "kom",
    "adt",
}

LEVELS = ("pos", "pos_case")

FOCUS = "*"  # prefix for the NE type of tokens in the focal span
MASKED = "NE"  # focal type after mask_focal_type, for type-blind contexts


@dataclass
class EgoGraph:
    """Ego graph of one NE span. Nodes are 0-based word indices."""

    text_id: int
    ne_index: int
    nertag: str
    ne_text: str
    labels: dict[int, str]  # node -> label
    edges: list[tuple[int, int, str]]  # (head, dependent, deprel)


def _case(form: str) -> str | None:
    found = [part for part in form.split() if part in CASES]
    return found[-1] if found else None


def _token_labels(sentence: dict, focal: set[int], level: str) -> list[str]:
    """Label every word: NE type (focal span marked with FOCUS), POS and optionally case.

    POS and case come from the first `morph_analysis` analysis, which keeps the
    proper noun tag H that the Stanza layer folds into S.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    ne_of = ["O"] * len(sentence["words"])
    for tag, idx in zip(sentence["ne_tag"], sentence["ne_words"]):
        for i in idx:
            ne_of[i] = tag

    labels = []
    for i, (pos, form) in enumerate(zip(sentence["pos"], sentence["form"])):
        parts = [(FOCUS + ne_of[i]) if i in focal else ne_of[i], pos]
        if level == "pos_case":
            parts.append(_case(form) or "-")
        labels.append(":".join(parts))
    return labels


def _nearest(candidates: Iterable[int], span: set[int], k: int) -> set[int]:
    """The k candidates closest to the span on each side (by word position)."""
    first, last = min(span), max(span)
    left = sorted((i for i in candidates if i < first), reverse=True)[:k]
    right = sorted(i for i in candidates if i > last)[:k]
    return set(left) | set(right)


def ego_graphs(
    sentence: dict, level: str = "pos_case", max_dependents: int = 2
) -> Iterator[EgoGraph]:
    """Yield one ego graph per NE span in a sentence record (see data.SENTENCE_SCHEMA).

    Nodes: span words, their dependents, the span's external head(s), the parent
    of each external head, and the other dependents of each external head.
    Dependents outside the span are capped at the `max_dependents` nearest to the
    span on each side, per head; without the cap, heads with long lists of
    dependents make gSpan's embedding count (and memory) explode.
    Edges: every dependency arc between two included nodes.
    """
    # Stanza ids are 1-based; convert heads to 0-based indices, root -> None
    heads = [h - 1 if h else None for h in sentence["head"]]
    deprels = sentence["deprel"]
    children = collections.defaultdict(list)
    for i, h in enumerate(heads):
        if h is not None:
            children[h].append(i)

    for ne_index, (tag, idx) in enumerate(
        zip(sentence["ne_tag"], sentence["ne_words"])
    ):
        span = set(idx)
        ext_heads = {
            heads[i] for i in span if heads[i] is not None and heads[i] not in span
        }
        nodes = set(span)
        span_deps = {c for i in span for c in children[i] if c not in span}
        nodes.update(_nearest(span_deps, span, max_dependents))
        for h in ext_heads:
            nodes.add(h)
            nodes.update(_nearest(set(children[h]) - span, span, max_dependents))
            if heads[h] is not None:
                nodes.add(heads[h])

        labels = _token_labels(sentence, span, level)
        yield EgoGraph(
            text_id=sentence["text_id"],
            ne_index=ne_index,
            nertag=tag,
            ne_text=" ".join(sentence["words"][i] for i in sorted(span)),
            labels={i: labels[i] for i in nodes},
            edges=[(heads[i], i, deprels[i]) for i in nodes if heads[i] in nodes],
        )


def mask_focal_type(ego: EgoGraph) -> EgoGraph:
    """Replace the focal span's NE type with MASKED (*PER:H:n -> *NE:H:n).

    Patterns mined from masked graphs are type-blind contexts; their support is
    the denominator of a typed pattern's confidence.
    """
    labels = {
        i: FOCUS + MASKED + label[label.index(":") :]
        if label.startswith(FOCUS)
        else label
        for i, label in ego.labels.items()
    }
    return dataclasses.replace(ego, labels=labels)


_DFS_EDGE = re.compile(r"\(frm=(\d+), to=(\d+), vevlb=(\(.*?\))\)")


def pattern_graph(dfs_code: str, mask: bool = False) -> nx.Graph:
    """The pattern as a labelled networkx graph, from its `dfs_code` string.

    With `mask`, the focal type is replaced by MASKED.
    """
    g = nx.Graph()
    for frm, to, vevlb in _DFS_EDGE.findall(dfs_code):
        vlb1, elb, vlb2 = ast.literal_eval(vevlb)
        for v, label in ((int(frm), vlb1), (int(to), vlb2)):
            if label != _gspan.VACANT_VERTEX_LABEL:
                if mask and label.startswith(FOCUS):
                    label = FOCUS + MASKED + label[label.index(":") :]
                g.add_node(v, label=label)
        g.add_edge(int(frm), int(to), label=elb)
    return g


def pattern_key(dfs_code: str, mask: bool = False) -> str:
    """Isomorphism-invariant key of a pattern, from its `dfs_code` string.

    Patterns are trees, which Weisfeiler-Lehman refinement identifies up to
    isomorphism, so equal keys mean equal patterns. With `mask`, the focal type
    is replaced by MASKED first, so a typed pattern maps to its context's key.
    """
    g = pattern_graph(dfs_code, mask)
    return nx.weisfeiler_lehman_graph_hash(
        g, node_attr="label", edge_attr="label", iterations=g.number_of_nodes()
    )


def contains(big: nx.Graph, small: nx.Graph) -> bool:
    """Whether pattern `small` is a sub-pattern of `big` (labels must match)."""
    same = lambda a, b: a["label"] == b["label"]
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        big, small, node_match=same, edge_match=same
    )
    return matcher.subgraph_is_monomorphic()


def _to_gspan_graph(gid: int, ego: EgoGraph) -> Graph:
    """Subdivide each arc into head -h- [dep:deprel] -d- dependent."""
    g = Graph(gid, is_undirected=True, eid_auto_increment=True)
    vid = {node: k for k, node in enumerate(sorted(ego.labels))}
    for node, label in ego.labels.items():
        g.add_vertex(vid[node], label)
    next_vid = len(vid)
    for head, dep, deprel in ego.edges:
        g.add_vertex(next_vid, f"dep:{deprel}")
        g.add_edge(AUTO_EDGE_ID, vid[head], next_vid, "h")
        g.add_edge(AUTO_EDGE_ID, next_vid, vid[dep], "d")
        next_vid += 1
    return g


def describe(vertices: dict[int, str], edges: list[tuple[int, int, str]]) -> str:
    """Readable pattern: one `head -deprel-> dependent` arc per relation node.

    Relation nodes missing their head or dependent are shown with `?`.
    """
    rel_ends = collections.defaultdict(dict)  # relation vid -> {"h": vid, "d": vid}
    for a, b, elb in edges:
        rel, word = (a, b) if vertices[a].startswith("dep:") else (b, a)
        rel_ends[rel][elb] = word
    arcs = []
    for rel, ends in rel_ends.items():
        head = vertices[ends["h"]] if "h" in ends else "?"
        dep = vertices[ends["d"]] if "d" in ends else "?"
        arcs.append(f"{head} -{vertices[rel][4:]}-> {dep}")
    return " ; ".join(sorted(arcs))


class _GSpan(_gspan.gSpan):
    """gSpan over in-memory graphs that collects patterns instead of printing them.

    The upstream `_report` uses `DataFrame.append`, removed in pandas 2.
    """

    def __init__(self, graphs: dict[int, Graph], focal_only: bool, **kwargs):
        super().__init__(database_file_name=None, **kwargs)
        self._input_graphs = graphs
        self._focal_only = focal_only
        self.patterns: list[dict] = []

    def _read_graphs(self):
        self.graphs = self._input_graphs
        return self

    def _report_size1(self, g, support):
        pass

    def _report(self, projected):
        n_vertices = self._DFScode.get_num_vertices()
        if n_vertices < self._min_num_vertices:
            return
        vertices, edges = {}, []
        for e in self._DFScode:
            vlb1, elb, vlb2 = e.vevlb
            if vlb1 != _gspan.VACANT_VERTEX_LABEL:
                vertices[e.frm] = vlb1
            if vlb2 != _gspan.VACANT_VERTEX_LABEL:
                vertices[e.to] = vlb2
            edges.append((e.frm, e.to, elb))
        # Filtering only the report keeps gSpan's search (and its supports) unchanged
        if self._focal_only and not any(v.startswith(FOCUS) for v in vertices.values()):
            return
        self.patterns.append(
            {
                "support": self._support,
                "n_vertices": n_vertices,
                "pattern": describe(vertices, edges),
                # every relation node has both its head and its dependent
                "complete": len(edges)
                == 2 * sum(v.startswith("dep:") for v in vertices.values()),
                "dfs_code": repr(self._DFScode),
                "graph_ids": sorted({p.gid for p in projected}),
            }
        )


def mine(
    egos: Iterable[EgoGraph],
    min_support: int,
    min_vertices: int = 3,
    max_vertices: int = 7,
    focal_only: bool = True,
) -> pd.DataFrame:
    """Run gSpan on ego graphs; return one row per frequent pattern.

    Vertex counts include relation nodes, so a pattern with k words and k-1
    arcs has 2k-1 vertices (3 = one arc, 7 = three arcs).
    `graph_ids` index into the list of ego graphs passed in.
    With `focal_only`, only patterns that include a focal-span word are returned.
    """
    egos = list(egos)
    graphs = {gid: _to_gspan_graph(gid, ego) for gid, ego in enumerate(egos)}
    miner = _GSpan(
        graphs,
        focal_only=focal_only,
        min_support=min_support,
        min_num_vertices=min_vertices,
        max_num_vertices=max_vertices,
    )
    miner.run()
    patterns = pd.DataFrame(
        miner.patterns,
        columns=[
            "support",
            "n_vertices",
            "complete",
            "pattern",
            "dfs_code",
            "graph_ids",
        ],
    )
    return patterns.sort_values("support", ascending=False, ignore_index=True)
