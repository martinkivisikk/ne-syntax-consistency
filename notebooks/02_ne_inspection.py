import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    from ne_syntax_consistency.data import LAYERS, load_sample

    return LAYERS, load_sample, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NE kihi esmane ülevaade

    Kollektsioon `koondkorpus_sentences_test_5000_sg_thread`, kihid `v171_named_entities`, `v172_stanza_syntax`, `morph_analysis`.

    1. Märgendid ja nende sagedused, spannide pikkused
    2. Joondumine: kas NE spannid algavad/lõpevad sõna- ja Stanza tokenite piiril
    3. Sidusus: kas NE moodustab ühe alampuu, millel on üks väline ülem
    4. NE peasõna deprel ja viimase tokeni kääne NE tüübi kaupa
    """)
    return


@app.cell
def _(mo):
    COLLECTION = "koondkorpus_sentences_test_5000_sg_thread"
    # Kogu kollektsioon on ~160k lauset (~11 min laadimist); vaikimisi võtame osa
    n_sentences = mo.ui.number(start=1000, stop=160000, step=1000, value=20000, label="Lauseid")
    n_sentences
    return COLLECTION, n_sentences


@app.cell
def _(COLLECTION, LAYERS, load_sample, mo, n_sentences):
    # Tulemus puhverdatakse kettale (__marimo__/cache), et korduv käivitus ei läheks andmebaasi
    with mo.persistent_cache(name=f"ne_sample_{n_sentences.value}"):
        texts = load_sample(n_sentences.value, LAYERS + ["words"], collection=COLLECTION)
    mo.md(f"Laaditud **{len(texts)}** lauset")
    return (texts,)


@app.cell
def _(pd, texts):
    CASES = {"nom", "gen", "part", "ill", "in", "el", "all", "ad", "abl", "tr", "ter", "es", "ab", "kom", "adit"}


    def _case(feats):
        _c = sorted(CASES & set(feats))
        return _c[0] if _c else None


    def _span_rows(text_id, text):
        words, syn = text["words"], text["v172_stanza_syntax"]
        start2i = {w.start: i for i, w in enumerate(words)}
        tokens = [s.annotations[0] for s in syn]
        heads = [tok["head"] for tok in tokens]
        for ne in text["v171_named_entities"]:
            idx = [start2i.get(w.start) for w in ne.spans]
            # Stanza id-d on 1-põhised, sõnade järjekord sama mis süntaksikihis
            ids = {i + 1 for i in idx if i is not None}
            roots = sorted(i for i in ids if heads[i - 1] not in ids)
            # Täielik alampuu: NE = juure kõik järglased (+ juur)
            subtree = set(roots)
            frontier = list(roots)
            while frontier:
                _h = frontier.pop()
                _kids = [j + 1 for j, h in enumerate(heads) if h == _h]
                subtree.update(_kids)
                frontier.extend(_kids)
            root = tokens[roots[0] - 1] if len(roots) == 1 else None
            last = tokens[max(ids) - 1]
            yield {
                "text_id": text_id,
                "nertag": ne.annotations[0]["nertag"],
                "ne_text": ne.enclosing_text,
                "n_words": len(ne.spans),
                "start": ne.start,
                "end": ne.end,
                # Joondumine
                "words_aligned": None not in idx,
                "syntax_aligned": len(syn) == len(words)
                and all(syn[i].start == words[i].start and syn[i].end == words[i].end for i in idx if i is not None),
                "contiguous": idx == list(range(idx[0], idx[-1] + 1)),
                # Sidusus
                "n_roots": len(roots),
                "complete_subtree": len(roots) == 1 and subtree == ids,
                "head_deprel": root["deprel"] if root else None,
                "head_upostag": root["upostag"] if root else None,
                "ext_head_lemma": tokens[root["head"] - 1]["lemma"] if root and root["head"] else None,
                "last_case": _case(last["feats"]),
                "last_upostag": last["upostag"],
                "deprels": " ".join(f"{tokens[i - 1]['deprel']}>{heads[i - 1]}" for i in sorted(ids)),
                "sentence": text.text,
            }


    spans = pd.DataFrame([r for text_id, text in texts for r in _span_rows(text_id, text)])
    spans
    return (spans,)


@app.cell
def _(mo, pd, spans):
    # 1. Märgendid ja pikkused (sõnades)
    mo.vstack([
        mo.md("## 1. Märgendid ja spannide pikkused"),
        spans["nertag"].value_counts().rename("spans").to_frame(),
        mo.md("Spanni pikkus sõnades (6 = 6+):"),
        pd.crosstab(spans["nertag"], spans["n_words"].clip(upper=6).astype(str), margins=True),
    ])
    return


@app.cell
def _(mo, spans):
    # 2. Joondumine. NE kiht ümbritseb `words` kihti ja süntaksikiht on `morph_extended` -> `words` peal,
    # seega peaks joondumine kehtima ehituslikult; kontrollime siiski andmetes üle
    _flags = ["words_aligned", "syntax_aligned", "contiguous"]
    _misaligned = spans[~spans[_flags].all(axis=1)]
    mo.vstack([
        mo.md("## 2. Joondumine"),
        (~spans[_flags]).sum().rename("rikkumisi").to_frame(),
        _misaligned[["nertag", "ne_text", "sentence"] + _flags].head(20) if len(_misaligned) else mo.md("Rikkumisi ei leitud."),
    ])
    return


@app.cell
def _(mo, pd, spans):
    # 3. Sidusus: n_roots = NE tokenite arv, mille ülem on väljaspool NE-d.
    # n_roots == 1 -> NE on üks sidus alampuu ühe välise ülemaga.
    # complete_subtree lisaks: juure kõik järglased kuuluvad NE-sse (rangem, nt "Eesti president" puhul ei kehti)
    _by_type = spans.groupby("nertag").agg(
        spans=("n_roots", "size"),
        single_root=("n_roots", lambda s: (s == 1).mean()),
        complete_subtree=("complete_subtree", "mean"),
    )
    _multi = spans[spans["n_words"] > 1].groupby("nertag").agg(
        multiword_spans=("n_roots", "size"),
        multiword_single_root=("n_roots", lambda s: (s == 1).mean()),
    )
    mo.vstack([
        mo.md("## 3. Sidusus"),
        _by_type.join(_multi).round(3),
        pd.crosstab(spans["nertag"], spans["n_roots"].astype(str).rename("n_roots")),
    ])
    return


@app.cell
def _(mo, spans):
    # Näited NE-dest, mis ei moodusta ühte alampuud (deprels: deprel>ülema id iga NE tokeni kohta)
    violations = spans[spans["n_roots"] > 1]
    mo.vstack([
        mo.md(f"**{len(violations)}** mitmejuurelist spanni; juhuslik valim:"),
        violations.sample(min(15, len(violations)), random_state=0)[["nertag", "ne_text", "n_roots", "deprels", "sentence"]],
    ])
    return


@app.cell
def _(mo, pd, spans):
    # 4. NE peasõna deprel ja viimase tokeni kääne NE tüübi kaupa (ainult ühe juurega spannid peasõna jaoks)
    def _top(col, k=12):
        _tab = pd.crosstab(spans[col].fillna("-"), spans["nertag"], normalize="columns")
        return _tab.loc[_tab.max(axis=1).sort_values(ascending=False).index[:k]].round(3)

    mo.vstack([
        mo.md("## 4. Peasõna deprel ja viimase tokeni kääne (osakaal NE tüübi sees)"),
        mo.hstack([_top("head_deprel"), _top("last_case", 15)], justify="start", gap=2),
    ])
    return


if __name__ == "__main__":
    app.run()
