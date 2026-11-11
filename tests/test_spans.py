from ne_syntax_consistency.spans import span_tree, span_trees


def _sentence(words, pos, form, head, deprel, ne_tag, ne_words):
    return {
        "text_id": 0,
        "words": words,
        "pos": pos,
        "form": form,
        "head": head,
        "deprel": deprel,
        "ne_tag": ne_tag,
        "ne_words": ne_words,
    }


def test_flat_name_and_word_order():
    # Mart Laar ütles : Laar -flat-> Mart, Mart -nsubj-> ütles
    s = _sentence(
        ["Mart", "Laar", "ütles"],
        ["H", "H", "V"],
        ["sg n", "sg n", "s"],
        [3, 1, 0],
        ["nsubj", "flat", "root"],
        ["PER"],
        [[0, 1]],
    )
    (row,) = span_trees(s)
    assert row["pattern"] == "PER [flat PER]"
    assert row["ext_deprel"] == "nsubj"
    assert span_tree(s, [0, 1], "PER", "pos_case")["pattern"] == "H:n [flat H:n]"


def test_left_dependents_and_forest():
    # Eesti Posti osakond : chain of genitive nmods to the left of the head
    s = _sentence(
        ["Eesti", "Posti", "osakond", "ja", "Tartu"],
        ["H", "H", "S", "J", "H"],
        ["sg g", "sg g", "sg n", "", "sg n"],
        [2, 3, 0, 5, 3],
        ["nmod", "nmod", "root", "cc", "conj"],
        ["ORG"],
        [[0, 1, 2]],
    )
    assert span_tree(s, [0, 1, 2], "ORG")["pattern"] == "[nmod [nmod ORG] ORG] ORG"
    # Tartu's head (osakond) is outside the span {Posti, Tartu}: two roots
    row = span_tree(s, [1, 4], "ORG")
    assert row["pattern"] == "ORG + ORG"
    assert (row["n_roots"], row["ext_deprel"]) == (2, "nmod+conj")


def test_top_patterns_and_sample_examples():
    import pandas as pd

    from ne_syntax_consistency.spans import (
        sample_examples,
        support_table,
        top_patterns,
    )

    spans = pd.DataFrame(
        {
            "text_id": range(9),
            "ne_index": 0,
            "nertag": ["PER"] * 6 + ["LOC"] * 3,
            "pattern": ["A"] * 4 + ["B"] * 2 + ["C"] * 3,
            "n_words": 1,
            "n_roots": 1,
            "ne_text": ["x", "x", "x", "y"] + ["z"] * 5,
        }
    )
    top = top_patterns(support_table(spans), min_support=2, top_n=1)
    assert top[["nertag", "pattern", "rank"]].values.tolist() == [
        ["LOC", "C", 1],
        ["PER", "A", 1],
    ]
    ex = sample_examples(spans, top, n=3, seed=1, distinct=False)
    assert ex.groupby("pattern").size().to_dict() == {"A": 3, "C": 3}
    assert set(ex[ex["pattern"] == "A"]["text_id"]) <= {0, 1, 2, 3}
    assert ex.equals(sample_examples(spans, top, n=3, seed=1, distinct=False))
    # distinct: one example per span text, with that text's count in the pattern
    ex = sample_examples(spans, top, n=3, seed=1)
    a = ex[ex["pattern"] == "A"].set_index("ne_text")["ne_text_count"].to_dict()
    assert a == {"x": 3, "y": 1}
    assert ex[ex["pattern"] == "C"]["ne_text_count"].tolist() == [3]


def test_merge_support_tables_matches_support_table():
    import pandas as pd

    from ne_syntax_consistency.spans import merge_support_tables, support_table

    spans = pd.DataFrame(
        {
            "nertag": ["PER", "PER", "LOC", "PER", "LOC", "ORG"],
            "pattern": ["PER", "PER [flat PER]", "LOC", "PER", "LOC", "ORG"],
            "n_words": [1, 2, 1, 1, 1, 1],
            "n_roots": [1, 1, 1, 1, 1, 1],
            "ne_text": ["Laar", "Mart Laar", "Tartu", "Kask", "Pärnu", "Delfi"],
        }
    )
    merged = merge_support_tables([support_table(spans[:3]), support_table(spans[3:])])
    pd.testing.assert_frame_equal(merged, support_table(spans))
