from ne_syntax_consistency.viz import CONTEXT_ARC, SPAN_ARC, dependency_svg

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


def test_highlights_internal_and_context_arcs():
    svg = dependency_svg(SENTENCE, [2, 3])
    assert svg.count(f'stroke="{SPAN_ARC}"') == 1  # flat
    assert svg.count(f'stroke="{CONTEXT_ARC}"') == 1  # nsubj
    assert "H:n" in svg and ">PER<" in svg


def test_window_skips_words():
    svg = dependency_svg(SENTENCE, [3], window=0)
    # Laar's head (Mart) is kept, Eile and ütles become a gap
    assert ">Mart<" in svg and ">Eile<" not in svg and "…" in svg
