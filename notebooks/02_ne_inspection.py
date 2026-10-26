import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import pyarrow.parquet as pq

    from ne_syntax_consistency.gspan import CASES

    DATA = Path(__file__).resolve().parents[1] / "data"
    return CASES, DATA, mo, pd, pq


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NE kihi esmane ülevaade

    Juhuslik lausevalim kollektsioonist `koondkorpus_sentences`, mis on eksporditud käsuga
    `scripts/export_sample.py` (vaikimisi `data/sample_200000_s1.parquet`, ~200k lauset).
    Kihid: `v171_named_entities`, `v172_stanza_syntax` (ülem, deprel) ja `morph_analysis`
    (sõnaliik, kääne ja lemma esimesest analüüsist).

    0. Ülevaade: lausete ja NE-de osakaalud, mitmejuureliste spannide osakaal
    1. Märgendid ja nende sagedused, spannide pikkused
    2. Sidusus: kas NE moodustab ühe alampuu, millel on üks väline ülem
    3. NE peasõna deprel ja viimase sõna kääne NE tüübi kaupa

    NB! Valimisse satuvad ainult laused, millel on NE kihi rida (~19,0 mln lauset 21,4 mln-st),
    seega NE-ga lausete osakaal kehtib nende lausete kohta.
    """)
    return


@app.cell
def _(mo):
    # Andmebaasist eksportimine käib ainult nupuvajutusel (~20 min, vajab .env-i KK_* seadeid)
    export_button = mo.ui.run_button(label="Ekspordi 200k valim andmebaasist (~20 min)")
    return (export_button,)


@app.cell
def _(DATA, export_button, mo):
    from ne_syntax_consistency.data import export_sample

    DEFAULT_SAMPLE = "sample_200000_s1.parquet"
    if export_button.value and not (DATA / DEFAULT_SAMPLE).exists():
        DATA.mkdir(exist_ok=True)
        with mo.status.spinner(title="Ekspordin valimit andmebaasist ...") as _spinner:
            for _written, _skipped in export_sample(DATA / DEFAULT_SAMPLE, n=200000, seed=1):
                _spinner.update(subtitle=f"{_written} lauset kirjutatud, {_skipped} vahele jäetud")

    _files = sorted(p.name for p in DATA.glob("sample_*.parquet"))
    mo.stop(
        not _files,
        mo.vstack(
            [
                mo.md(
                    f"""
                    **Kaustas `{DATA}` pole ühtegi valimit.** Vajuta nuppu, et eksportida
                    200k juhuslikku lauset (seed 1) andmebaasist, või käivita terminalis:

                    ```
                    uv run python scripts/export_sample.py --sentences 200000 --seed 1
                    ```
                    """
                ),
                export_button,
            ]
        ),
    )
    sample_file = mo.ui.dropdown(
        _files,
        value=DEFAULT_SAMPLE if DEFAULT_SAMPLE in _files else _files[0],
        label="Valim",
    )
    sample_file
    return (sample_file,)


@app.cell
def _(DATA, mo, pq, sample_file):
    sentences = pq.read_table(DATA / sample_file.value).to_pylist()
    mo.md(f"Laaditud **{len(sentences)}** lauset failist `{sample_file.value}`")
    return (sentences,)


@app.cell
def _(CASES, pd, sentences):
    def _case(form):
        _c = [p for p in form.split() if p in CASES]
        return _c[-1] if _c else None


    def _span_rows(s):
        # Stanza ülemad on 1-põhised (0 = juur); teisendame 0-põhiseks, juur -> None
        heads = [h - 1 if h else None for h in s["head"]]
        children = {}
        for _i, _h in enumerate(heads):
            children.setdefault(_h, []).append(_i)
        for tag, idx in zip(s["ne_tag"], s["ne_words"]):
            ids = set(idx)
            roots = sorted(i for i in ids if heads[i] not in ids)
            # Täielik alampuu: NE = juure kõik järglased (+ juur)
            subtree = set(roots)
            frontier = list(roots)
            while frontier:
                _kids = children.get(frontier.pop(), [])
                subtree.update(_kids)
                frontier.extend(_kids)
            root = roots[0] if len(roots) == 1 else None
            ext_head = heads[root] if root is not None else None
            last = max(ids)
            yield {
                "text_id": s["text_id"],
                "nertag": tag,
                "ne_text": " ".join(s["words"][i] for i in sorted(ids)),
                "n_words": len(ids),
                # Sidusus
                "n_roots": len(roots),
                "complete_subtree": len(roots) == 1 and subtree == ids,
                "head_deprel": s["deprel"][root] if root is not None else None,
                "head_pos": s["pos"][root] if root is not None else None,
                "ext_head_lemma": s["lemma"][ext_head] if ext_head is not None else None,
                "last_case": _case(s["form"][last]),
                "last_pos": s["pos"][last],
                "deprels": " ".join(
                    f"{s['deprel'][i]}>{s['head'][i]}" for i in sorted(ids)
                ),
                "sentence": s["text"],
            }


    spans = pd.DataFrame([r for s in sentences for r in _span_rows(s)])
    spans
    return (spans,)


@app.cell
def _(mo, pd, sentences, spans):
    # 0. Ülevaade: osakaalud kogu valimi ja NE tüüpide kaupa
    _n_sent = len(sentences)
    _with_ne = sum(1 for s in sentences if s["ne_tag"])
    _multi = spans["n_words"] > 1
    _multiroot = spans["n_roots"] > 1
    _overview = pd.DataFrame(
        {
            "väärtus": [
                _n_sent,
                _with_ne,
                len(spans),
                int(_multi.sum()),
                int(_multiroot.sum()),
            ],
            "osakaal": [
                "",
                f"{_with_ne / _n_sent:.1%} lausetest",
                f"{len(spans) / _with_ne:.2f} NE-d NE-ga lause kohta",
                f"{_multi.mean():.1%} NE spannidest",
                (
                    f"{_multiroot.mean():.1%} kõigist spannidest, "
                    f"{_multiroot[_multi].mean():.1%} mitmesõnalistest"
                ),
            ],
        },
        index=[
            "lauseid",
            "lauseid, kus on ≥1 NE",
            "NE spanne",
            "mitmesõnalisi spanne",
            "mitmejuurelisi spanne",
        ],
    )

    _g = spans.assign(multi=_multi, multiroot=_multiroot).groupby("nertag")
    _by_type = pd.DataFrame(
        {
            "spanne": _g.size(),
            "% spannidest": _g.size() / len(spans) * 100,
            "% mitmesõnalisi": _g["multi"].mean() * 100,
            "mitmejuurelisi": _g["multiroot"].sum(),
            "% mitmejuurelisi (kõigist)": _g["multiroot"].mean() * 100,
            "% mitmejuurelisi (mitmesõnalistest)": _g.apply(
                lambda d: d.loc[d["multi"], "multiroot"].mean() * 100,
                include_groups=False,
            ),
            "% täielik alampuu (mitmesõnalistest)": _g.apply(
                lambda d: d.loc[d["multi"], "complete_subtree"].mean() * 100,
                include_groups=False,
            ),
        }
    ).round(1)
    _sent_types = pd.Series(
        {
            _t: sum(1 for s in sentences if _t in s["ne_tag"]) / _n_sent * 100
            for _t in sorted(spans["nertag"].unique())
        },
        name="% lausetest, kus see tüüp esineb",
    ).round(1)

    mo.vstack(
        [
            mo.md("## 0. Ülevaade"),
            _overview,
            mo.md("NE tüüpide kaupa (protsentides):"),
            _by_type.join(_sent_types),
        ]
    )
    return


@app.cell
def _(mo, pd, spans):
    # 1. Märgendid ja pikkused (sõnades)
    mo.vstack(
        [
            mo.md("## 1. Märgendid ja spannide pikkused"),
            spans["nertag"].value_counts().rename("spans").to_frame(),
            mo.md("Spanni pikkus sõnades (6 = 6+):"),
            pd.crosstab(
                spans["nertag"],
                spans["n_words"].clip(upper=6).astype(str),
                margins=True,
            ),
        ]
    )
    return


@app.cell
def _(mo, pd, spans):
    # 2. Sidusus: n_roots = NE sõnade arv, mille ülem on väljaspool NE-d.
    # n_roots == 1 -> NE on üks sidus alampuu ühe välise ülemaga.
    # complete_subtree lisaks: juure kõik järglased kuuluvad NE-sse (rangem, nt "Eesti president" puhul ei kehti)
    _by_type = spans.groupby("nertag").agg(
        spans=("n_roots", "size"),
        single_root=("n_roots", lambda s: (s == 1).mean()),
        complete_subtree=("complete_subtree", "mean"),
    )
    _multi = (
        spans[spans["n_words"] > 1]
        .groupby("nertag")
        .agg(
            multiword_spans=("n_roots", "size"),
            multiword_single_root=("n_roots", lambda s: (s == 1).mean()),
        )
    )
    mo.vstack(
        [
            mo.md("## 2. Sidusus"),
            _by_type.join(_multi).round(3),
            pd.crosstab(spans["nertag"], spans["n_roots"].astype(str).rename("n_roots")),
        ]
    )
    return


@app.cell
def _(mo, spans):
    # Näited NE-dest, mis ei moodusta ühte alampuud (deprels: deprel>ülema id iga NE sõna kohta)
    violations = spans[spans["n_roots"] > 1]
    mo.vstack(
        [
            mo.md(f"**{len(violations)}** mitmejuurelist spanni; juhuslik valim:"),
            violations.sample(min(15, len(violations)), random_state=0)[
                ["nertag", "ne_text", "n_roots", "deprels", "sentence"]
            ],
        ]
    )
    return


@app.cell
def _(mo, pd, spans):
    # 3. NE peasõna deprel ja viimase sõna kääne NE tüübi kaupa (ainult ühe juurega spannid peasõna jaoks).
    # Kääne ja sõnaliik on nüüd morph_analysis'est (Vabamorfi koodid: n, g, in, ...), mitte Stanza feats'ist
    def _top(col, k=12):
        _tab = pd.crosstab(spans[col].fillna("-"), spans["nertag"], normalize="columns")
        return _tab.loc[_tab.max(axis=1).sort_values(ascending=False).index[:k]].round(3)

    mo.vstack(
        [
            mo.md("## 3. Peasõna deprel ja viimase sõna kääne (osakaal NE tüübi sees)"),
            mo.hstack(
                [_top("head_deprel"), _top("last_case", 15)], justify="start", gap=2
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
