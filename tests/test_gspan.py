from ne_syntax_consistency.gspan import EgoGraph, describe, ego_graphs, mine


def _ego(gid, labels, edges):
    return EgoGraph(
        text_id=gid, ne_index=0, nertag="PER", ne_text="", labels=labels, edges=edges
    )


def test_describe_keeps_direction():
    vertices = {0: "*PER:H:n", 1: "dep:flat", 2: "*PER:H:n"}
    assert describe(vertices, [(0, 1, "h"), (1, 2, "d")]) == "*PER:H:n -flat-> *PER:H:n"
    assert describe({0: "O:V:-", 1: "dep:nsubj"}, [(0, 1, "h")]) == "O:V:- -nsubj-> ?"


def test_mine_counts_graphs_and_direction():
    # Two graphs where verb -> name (nsubj), one where the arc is reversed
    egos = [
        _ego(0, {0: "O:V:-", 1: "*PER:H:n"}, [(0, 1, "nsubj")]),
        _ego(
            1,
            {0: "O:V:-", 1: "*PER:H:n", 2: "*PER:H:n"},
            [(0, 1, "nsubj"), (1, 2, "flat")],
        ),
        _ego(2, {0: "O:V:-", 1: "*PER:H:n"}, [(1, 0, "nsubj")]),
    ]
    patterns = mine(egos, min_support=2, min_vertices=3, focal_only=False)
    by_pattern = dict(zip(patterns["pattern"], patterns["support"]))
    assert by_pattern["O:V:- -nsubj-> *PER:H:n"] == 2
    assert "*PER:H:n -nsubj-> O:V:-" not in by_pattern  # only once, below min_support
    assert not any("flat" in p for p in by_pattern)

    sub = patterns[patterns["pattern"] == "O:V:- -nsubj-> *PER:H:n"].iloc[0]
    assert sub["graph_ids"] == [0, 1]


def test_focal_only_and_complete():
    egos = [
        _ego(
            g, {0: "O:V:-", 1: "*PER:H:n", 2: "O:S:n"}, [(0, 1, "nsubj"), (0, 2, "obj")]
        )
        for g in range(2)
    ]
    patterns = mine(egos, min_support=2, min_vertices=2)
    assert all("*PER" in p for p in patterns["pattern"])
    complete = dict(zip(patterns["pattern"], patterns["complete"]))
    assert complete["O:V:- -nsubj-> *PER:H:n"]
    assert not complete["? -nsubj-> *PER:H:n"]


def _sentence(n_words, heads, ne_words, tag="PER"):
    return {
        "text_id": 1,
        "words": [f"w{i}" for i in range(n_words)],
        "pos": ["S"] * n_words,
        "form": ["sg n"] * n_words,
        "head": heads,
        "deprel": ["dep"] * n_words,
        "ne_tag": [tag],
        "ne_words": [ne_words],
    }


def test_ego_graph_nodes_and_cap():
    # Word 0 is the root verb with 9 dependents (1..9); the NE is words 4-5,
    # with 5 attached to 4 (flat). Only 2 siblings per side should be kept.
    heads = [0, 1, 1, 1, 1, 5, 1, 1, 1, 1]

    [ego] = ego_graphs(_sentence(10, heads, [4, 5]), max_dependents=2)
    assert set(ego.labels) == {0, 2, 3, 4, 5, 6, 7}
    assert (4, 5, "dep") in ego.edges and (0, 4, "dep") in ego.edges
    assert ego.labels[4] == "*PER:S:n" and ego.labels[0] == "O:S:n"
    assert ego.ne_text == "w4 w5"


def test_pattern_key_distinguishes_chain_and_siblings():
    from ne_syntax_consistency.gspan import mask_focal_type, pattern_key

    # A -nmod-> B -nmod-> B (chain) vs B <-nmod- A -nmod-> B (siblings): same labels
    chain = [
        _ego(
            g, {0: "O:S:n", 1: "*LOC:H:g", 2: "O:S:g"}, [(0, 1, "nmod"), (1, 2, "nmod")]
        )
        for g in range(2)
    ]
    siblings = [
        _ego(
            g, {0: "O:S:n", 1: "*LOC:H:g", 2: "O:S:g"}, [(0, 1, "nmod"), (0, 2, "nmod")]
        )
        for g in range(2)
    ]
    full = lambda egos: mine(egos, 2, min_vertices=5).query("complete").iloc[0]
    a, b = full(chain), full(siblings)
    assert pattern_key(a["dfs_code"]) != pattern_key(b["dfs_code"])

    # Masking the typed pattern gives the key of the pattern mined from masked graphs
    masked = full([mask_focal_type(e) for e in chain])
    assert "*NE:H:g" in masked["pattern"]
    assert pattern_key(a["dfs_code"], mask=True) == pattern_key(masked["dfs_code"])
    assert pattern_key(a["dfs_code"], mask=True) != pattern_key(a["dfs_code"])
