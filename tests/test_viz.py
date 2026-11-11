from ne_syntax_consistency.viz import record_model, record_svg

# Eile ütles Mart Laar : Laar -flat-> Mart -nsubj-> ütles
SENTENCE = {
    "words": ["Eile", "ütles", "Mart", "Laar"],
    "pos": ["D", "V", "H", "H"],
    "form": ["", "s", "sg n", "sg n"],
    "head": [2, 0, 2, 3],
    "deprel": ["advmod", "root", "nsubj", "flat"],
    "ne_tag": ["PER"],
    "ne_words": [[2, 3]],
}


def test_record_model():
    model = record_model(SENTENCE, [2, 3])
    assert model.heads == (1, None, 1, 2)  # 1-based heads -> 0-based, root -> None
    assert [(r.label, r.values) for r in model.rows] == [
        ("pos", ("D", "V", "H", "H")),
        ("case", ("-", "-", "n", "n")),
    ]
    assert [(b.start, b.end, b.label) for b in model.boxes] == [(2, 4, "PER")]
    assert model.highlight_roots == {2}


def test_highlights_internal_and_context_arcs():
    svg = record_svg(SENTENCE, [2, 3])
    assert svg.count('class="arc highlight"') == 1  # flat
    assert svg.count('class="arc context"') == 1  # nsubj
    assert ">PER<" in svg


def test_window_hides_words():
    svg = record_svg(SENTENCE, [3], window_size=0)
    # Laar's head (Mart) is kept, Eile and ütles become a gap
    assert ">Mart<" in svg and ">Eile<" not in svg and "…" in svg
