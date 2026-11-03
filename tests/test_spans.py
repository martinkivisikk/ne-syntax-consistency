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
