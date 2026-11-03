"""Dependency trees of sentence records as inline SVG arc diagrams.

Words stay in sentence order; every arc runs from head to dependent (arrow at
the dependent) with its deprel on top, and nested arcs are drawn higher. An NE
span can be highlighted: its words get a coloured box, the arcs inside the span
(the span's internal tree) are drawn thick, and the arcs from the span's roots
to their heads outside the span are dashed.
"""

from html import escape

from ne_syntax_consistency.gspan import _case

# Colours; the SVG has its own white background, so they work in dark mode too
SPAN_FILL = "#ffe3c2"
SPAN_ARC = "#d9480f"
CONTEXT_ARC = "#1c7ed6"
OTHER_NE_FILL = "#e7eef7"
ARC = "#9a9a9a"
TEXT = "#222"
MUTED = "#777"

CHAR_W = 7.4  # approximate width of a 13px character
PAD = 14  # horizontal padding per word
LEVEL_H = 26  # height per nesting level of arcs
MAX_LEVEL = 10
GAP_W = 24  # width of the "…" column for skipped words


def _levels(arcs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Nesting level of each arc (column pairs): 1 + the highest arc it covers."""
    levels = {}
    for a, b in sorted(arcs, key=lambda arc: abs(arc[0] - arc[1])):
        lo, hi = min(a, b), max(a, b)
        inner = [
            lvl for (c, d), lvl in levels.items() if lo <= min(c, d) and max(c, d) <= hi
        ]
        levels[(a, b)] = min(1 + max(inner, default=0), MAX_LEVEL)
    return levels


def dependency_svg(
    sentence: dict, span=(), window: int | None = None, labels: bool = True
) -> str:
    """SVG arc diagram of one sentence record (see data.SENTENCE_SCHEMA).

    `span`: 0-based word indices of the NE to highlight. `window`: if given,
    show only the span, `window` words on each side, and the heads of the span's
    roots; skipped stretches become "…". `labels` adds POS:case and NE type
    rows under the words.
    """
    words = sentence["words"]
    n = len(words)
    heads = [h - 1 if h else None for h in sentence["head"]]
    deprels = sentence["deprel"]
    span = set(span)
    ne_of = {}
    for tag, idx in zip(sentence["ne_tag"], sentence["ne_words"]):
        for i in idx:
            ne_of[i] = tag
    roots = {i for i in span if heads[i] not in span}

    if window is None or not span:
        shown = list(range(n))
    else:
        keep = set(range(max(min(span) - window, 0), min(max(span) + window + 1, n)))
        keep |= {heads[i] for i in roots if heads[i] is not None}
        shown = sorted(keep)

    # Columns: shown words, plus a gap column wherever words are skipped
    columns = []  # (word index or None for a gap, width)
    for k, i in enumerate(shown):
        if (k == 0 and i > 0) or (k > 0 and i > shown[k - 1] + 1):
            columns.append((None, GAP_W))
        sub = f"{sentence['pos'][i]}:{_case(sentence['form'][i]) or '-'}"
        text = max(len(words[i]), len(sub) if labels else 0, len(ne_of.get(i, "")))
        columns.append((i, text * CHAR_W + PAD))
    if shown and shown[-1] < n - 1:
        columns.append((None, GAP_W))

    x_of, centre, x = {}, [], 0.0
    for i, w in columns:
        centre.append(x + w / 2)
        if i is not None:
            x_of[i] = len(centre) - 1
        x += w
    width = x

    arcs = {
        (x_of[heads[i]], x_of[i]): i
        for i in shown
        if heads[i] is not None and heads[i] in x_of
    }
    levels = _levels(list(arcs))
    top = (max(levels.values(), default=0) + 1) * LEVEL_H + 10
    y_arc = top  # arcs start and end here
    y_word = top + 18
    height = y_word + (40 if labels else 8)

    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" font-family="sans-serif" font-size="13">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        "<defs>",
        *(
            f'<marker id="a{c[1:]}" viewBox="0 0 10 10" refX="10" refY="5" '
            f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
            for c in (ARC, SPAN_ARC, CONTEXT_ARC)
        ),
        "</defs>",
    ]

    # Word boxes
    for col, (i, w) in enumerate(columns):
        x0 = centre[col] - w / 2 + 2
        if i in span:
            fill = SPAN_FILL
        elif i in ne_of:
            fill = OTHER_NE_FILL
        else:
            continue
        out.append(
            f'<rect x="{x0:.1f}" y="{y_word - 14}" width="{w - 4:.1f}" '
            f'height="{height - y_word + 10:.0f}" rx="4" fill="{fill}"/>'
        )

    # Arcs, the highlighted ones last so they are drawn on top
    def style(h: int, d: int) -> tuple[str, str, str]:
        if d in span and h in span:
            return SPAN_ARC, "2.5", ""
        if d in roots:
            return CONTEXT_ARC, "2", ' stroke-dasharray="5,3"'
        return ARC, "1", ""

    order = sorted(
        arcs.items(), key=lambda item: style(heads[item[1]], item[1])[0] != ARC
    )
    for (ch, cd), d in order:
        colour, stroke, dash = style(heads[d], d)
        x1, x2 = centre[ch], centre[cd]
        # Small horizontal offsets keep arcs that meet at one word apart
        x1 += 3 if x2 > x1 else -3
        h = levels[(ch, cd)] * LEVEL_H
        out.append(
            f'<path d="M{x1:.1f},{y_arc} C{x1:.1f},{y_arc - h} {x2:.1f},{y_arc - h} '
            f'{x2:.1f},{y_arc}" fill="none" stroke="{colour}" stroke-width="{stroke}"'
            f'{dash} marker-end="url(#a{colour[1:]})"/>'
        )
        weight = ' font-weight="bold"' if colour != ARC else ""
        out.append(
            f'<text x="{(x1 + x2) / 2:.1f}" y="{y_arc - 0.75 * h - 3:.1f}" '
            f'text-anchor="middle" font-size="11" fill="{colour if colour != ARC else MUTED}"'
            f'{weight} paint-order="stroke" stroke="white" stroke-width="3">'
            f"{escape(deprels[d])}</text>"
        )

    # Root marker and words
    for col, (i, _) in enumerate(columns):
        cx = centre[col]
        if i is None:
            out.append(
                f'<text x="{cx:.1f}" y="{y_word}" text-anchor="middle" fill="{MUTED}">…</text>'
            )
            continue
        if heads[i] is None:
            out.append(
                f'<line x1="{cx:.1f}" y1="4" x2="{cx:.1f}" y2="{y_arc}" stroke="{ARC}" '
                f'marker-end="url(#a{ARC[1:]})"/>'
            )
        weight = ' font-weight="bold"' if i in span else ""
        out.append(
            f'<text x="{cx:.1f}" y="{y_word}" text-anchor="middle" fill="{TEXT}"'
            f"{weight}>{escape(words[i])}</text>"
        )
        if labels:
            sub = f"{sentence['pos'][i]}:{_case(sentence['form'][i]) or '-'}"
            out.append(
                f'<text x="{cx:.1f}" y="{y_word + 16}" text-anchor="middle" '
                f'font-size="11" fill="{MUTED}">{escape(sub)}</text>'
            )
            if i in ne_of:
                out.append(
                    f'<text x="{cx:.1f}" y="{y_word + 30}" text-anchor="middle" '
                    f'font-size="11" fill="{TEXT}">{ne_of[i]}</text>'
                )
    out.append("</svg>")
    return "\n".join(out)
