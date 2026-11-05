import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    from ne_syntax_consistency.spans import LABELS, span_trees, support_table
    from ne_syntax_consistency.viz import (
        CONTEXT_ARC,
        SPAN_ARC,
        SPAN_FILL,
        dependency_svg,
    )

    DATA = Path(__file__).resolve().parents[1] / "data"
    RESULTS = Path(__file__).resolve().parents[1] / "results"
    COLUMNS = [
        "text_id",
        "words",
        "pos",
        "form",
        "head",
        "deprel",
        "ne_tag",
        "ne_words",
    ]
    return (
        COLUMNS,
        CONTEXT_ARC,
        DATA,
        LABELS,
        RESULTS,
        SPAN_ARC,
        SPAN_FILL,
        dependency_svg,
        ds,
        mo,
        pd,
        pq,
        span_trees,
        support_table,
    )


@app.cell(hide_code=True)
def _(CONTEXT_ARC, SPAN_ARC, SPAN_FILL, mo):
    mo.md(f"""
    # NE spannide sisemised puud

    Iga NE span annab täpselt ühe mustri: tema sõnade ja nendevaheliste kaarte puu
    (`scripts/span_trees.py`, `ne_syntax_consistency.spans`). Mustri tugi = spannide arv,
    mille terve sisemine puu on just see muster; alammustreid ei loeta. Uurime mustreid,
    mille tugi on vähemalt piirväärtus (vaikimisi 10), ja iga mustri kohta juhuslikke näitelauseid.

    Joonisel: <span style="background:{SPAN_FILL};padding:0 4px">span</span>,
    <b style="color:{SPAN_ARC}">paks kaar</b> = spanni sisene kaar (muster),
    <b style="color:{CONTEXT_ARC}">katkendkaar</b> = spanni juure seos välise ülemaga.
    Teised NE-d on helesinisel taustal. Sõna all on sõnaliik:kääne (`morph_analysis`) ja NE tüüp.

    Valim `full` on kogu korpuse NE-ga laused (`data/full`, `scripts/export_full.py`).
    Selle spannid loetakse eelnevalt arvutatud failist
    (`uv run python scripts/span_trees.py data/full --label <märgend>`)
    ja lauseid laetakse ainult näidete jaoks.
    """)
    return


@app.cell
def _(DATA, LABELS, mo):
    _files = sorted(p.name for p in DATA.glob("sample_*.parquet"))
    if any((DATA / "full").glob("part_*.parquet")):
        _files.append("full")
    mo.stop(
        not _files,
        mo.md(
            f"**Kaustas `{DATA}` pole valimit.** Ekspordi see: "
            "`uv run python scripts/export_sample.py --sentences 200000 --seed 1`"
        ),
    )
    sample_file = mo.ui.dropdown(
        _files,
        value="sample_200000_s1.parquet"
        if "sample_200000_s1.parquet" in _files
        else _files[0],
        label="Valim",
    )
    label = mo.ui.dropdown(list(LABELS), value="type", label="Sõnade märgend mustris")
    min_support = mo.ui.number(start=1, value=10, label="Min tugi")
    top_n = mo.ui.slider(1, 50, value=10, label="Mustreid tüübi kohta", show_value=True)
    n_examples = mo.ui.slider(
        1, 10, value=3, label="Näiteid mustri kohta", show_value=True
    )
    window = mo.ui.slider(
        0, 15, value=5, label="Kontekstiaken (0 = terve lause)", show_value=True
    )
    seed = mo.ui.number(start=0, value=1, label="Seeme")
    mo.vstack(
        [
            mo.hstack([sample_file, label, min_support, seed], justify="start"),
            mo.hstack([top_n, n_examples, window], justify="start"),
        ]
    )
    return label, min_support, n_examples, sample_file, seed, top_n, window


@app.cell
def _(COLUMNS, DATA, ds, mo, pq, sample_file):
    if sample_file.value == "full":
        # kogu korpus ei mahu mällu: laeme ainult küsitud laused
        all_sentences = None
        _dataset = ds.dataset(DATA / "full")

        def get_sentences(ids):
            _table = _dataset.to_table(
                columns=COLUMNS, filter=ds.field("text_id").isin(sorted(set(ids)))
            )
            return {s["text_id"]: s for s in _table.to_pylist()}

    else:
        with mo.status.spinner(title=f"Loen {sample_file.value} ..."):
            all_sentences = {
                s["text_id"]: s
                for s in pq.read_table(
                    DATA / sample_file.value, columns=COLUMNS
                ).to_pylist()
            }

        def get_sentences(ids):
            return {i: all_sentences[i] for i in ids}

    return all_sentences, get_sentences


@app.cell
def _(
    RESULTS,
    all_sentences,
    label,
    mo,
    pd,
    sample_file,
    span_trees,
    support_table,
):
    if all_sentences is None:
        _path = RESULTS / f"span_trees_{label.value}_full_spans.parquet"
        mo.stop(
            not _path.exists(),
            mo.md(
                f"**`{_path.name}` puudub.** Arvuta see: "
                f"`uv run python scripts/span_trees.py data/full --label {label.value}`"
            ),
        )
        with mo.status.spinner(title=f"Loen {_path.name} ..."):
            spans = pd.read_parquet(
                _path,
                columns=[
                    "text_id",
                    "ne_index",
                    "nertag",
                    "pattern",
                    "n_words",
                    "n_roots",
                    "ne_text",
                ],
                dtype_backend="pyarrow",
            )
    else:
        spans = pd.DataFrame(
            row for s in all_sentences.values() for row in span_trees(s, label.value)
        )
    table = support_table(spans)
    # mustri spannide reaindeksid, et näiteid kiiresti valida
    pattern_rows = spans.groupby(["nertag", "pattern"]).indices
    return pattern_rows, spans, table


@app.cell
def _(min_support, mo, table, top_n):
    # Iga tüübi top-N mustrit, mille tugi >= min tugi
    top = (
        table[table["support"] >= min_support.value]
        .groupby("nertag", group_keys=False)
        .head(top_n.value)
    )
    _summary = (
        table.assign(frequent=table["support"] >= min_support.value)
        .groupby(["nertag", "frequent"])
        .agg(patterns=("pattern", "size"), spans=("support", "sum"))
        .unstack("frequent", fill_value=0)
    )
    mo.vstack(
        [
            mo.md(f"### Mustrite arv tüübi kaupa (tugi ≥ {min_support.value}: True)"),
            _summary,
            mo.ui.table(
                top[["nertag", "support", "share_of_type", "n_roots", "pattern"]],
                selection=None,
                page_size=15,
            ),
        ]
    )
    return (top,)


@app.cell
def _(
    dependency_svg,
    get_sentences,
    mo,
    n_examples,
    pattern_rows,
    seed,
    spans,
    window,
):
    def _example(row, sentences):
        s = sentences[row.text_id]
        span = s["ne_words"][row.ne_index]
        svg = dependency_svg(s, span, window=window.value or None)
        return mo.vstack(
            [
                mo.md(f"`{row.text_id}` · {' '.join(s['words'])}"),
                mo.Html(f'<div style="overflow-x:auto">{svg}</div>'),
            ],
            gap=0.25,
        )

    def _examples(p):
        matching = spans.iloc[pattern_rows[(p.nertag, p.pattern)]]
        return matching.sample(
            n=min(n_examples.value, len(matching)), random_state=seed.value
        )

    def _pattern_block(p, examples, sentences):
        return mo.vstack(
            [
                mo.md(
                    f"#### `{p.pattern}`\n\ntugi **{p.support}** "
                    f"({p.share_of_type:.1%} tüübi spannidest), juuri {p.n_roots}"
                ),
                *(_example(r, sentences) for r in examples.itertuples()),
            ]
        )

    def pattern_tabs(patterns):
        """Example sentences for each pattern, one tab per NE type."""
        examples = {(p.nertag, p.pattern): _examples(p) for p in patterns.itertuples()}
        sentences = get_sentences(
            [i for e in examples.values() for i in e["text_id"].tolist()]
        )
        return mo.ui.tabs(
            {
                tag: mo.vstack(
                    [
                        _pattern_block(p, examples[(p.nertag, p.pattern)], sentences)
                        for p in group.itertuples()
                    ],
                    gap=2,
                )
                for tag, group in patterns.groupby("nertag")
            }
        )

    return (pattern_tabs,)


@app.cell
def _(pattern_tabs, top):
    pattern_tabs(top)
    return


@app.cell(hide_code=True)
def _(min_support, mo):
    mo.md(f"""
    ## Harvad mustrid (tugi < {min_support.value})

    Harvad mustrid on kõige tõenäolisemad kohad, kus NE piirid või märgendus on
    vastuolus süntaksiga. Vali tabelist mustrid, mille näiteid vaadata; kui midagi pole
    valitud, näidatakse iga tüübi kohta valitud arv mustreid (kõige sagedasemad
    harvadest või juhuslikud, sama seemnega mis näitelaused).
    """)
    return


@app.cell
def _(mo):
    rare_order = mo.ui.dropdown(
        {"sagedasemad enne": "support", "juhuslik": "random"},
        value="sagedasemad enne",
        label="Järjestus",
    )
    rare_n = mo.ui.slider(
        1, 50, value=10, label="Mustreid tüübi kohta", show_value=True
    )
    mo.hstack([rare_order, rare_n], justify="start")
    return rare_n, rare_order


@app.cell
def _(min_support, mo, rare_n, rare_order, seed, table):
    _rare = table[table["support"] < min_support.value]
    if rare_order.value == "random":
        _rare = _rare.sample(frac=1, random_state=seed.value).sort_values(
            "nertag", kind="stable"
        )
    rare_default = _rare.groupby("nertag", group_keys=False).head(rare_n.value)
    rare_table = mo.ui.table(
        _rare[["nertag", "support", "share_of_type", "n_roots", "pattern", "example"]],
        selection="multi",
        page_size=15,
        label=f"{len(_rare)} harva mustrit",
    )
    rare_table
    return rare_default, rare_table


@app.cell
def _(pattern_tabs, rare_default, rare_table, table):
    _selected = rare_table.value
    pattern_tabs(
        table.merge(_selected[["nertag", "pattern"]])
        if len(_selected)
        else rare_default
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Mitme juurega spannid

    Mitme juurega spanni sõnad ei moodusta üht sidusat alampuud: mitu spanni sõna on
    seotud väljaspool spanni oleva ülemaga (mustris ` + `). See viitab kas NE piiri veale
    (nt kaks NE-d ühes spannis või puudu jäänud nimeosa) või süntaksianalüüsi veale.
    Vali tabelist mustrid, mille näiteid vaadata; kui midagi pole valitud, näidatakse iga
    tüübi kohta valitud arv kõige sagedasemaid mustreid (ka harvu).
    """)
    return


@app.cell
def _(mo, pd, table):
    # Mitme juurega spannide osakaal tüübi kaupa (kõigist ja mitmesõnalistest spannidest)
    _t = table.assign(multi=table["n_roots"] > 1, multi_word=table["n_words"] > 1)
    _summary = pd.DataFrame(
        {
            "spans": _t.groupby("nertag")["support"].sum(),
            "multi_word": _t[_t["multi_word"]].groupby("nertag")["support"].sum(),
            "multi_root": _t[_t["multi"]].groupby("nertag")["support"].sum(),
        }
    ).fillna(0)
    _summary["share"] = _summary["multi_root"] / _summary["spans"]
    _summary["share_of_multi_word"] = _summary["multi_root"] / _summary["multi_word"]
    multi_n = mo.ui.slider(
        1, 50, value=10, label="Mustreid tüübi kohta", show_value=True
    )
    mo.vstack(
        [
            mo.ui.table(
                _summary.reset_index(),
                selection=None,
                format_mapping={
                    "share": "{:.2%}".format,
                    "share_of_multi_word": "{:.1%}".format,
                },
            ),
            multi_n,
        ]
    )
    return (multi_n,)


@app.cell
def _(mo, multi_n, table):
    _multi = table[table["n_roots"] > 1]
    multi_default = _multi.groupby("nertag", group_keys=False).head(multi_n.value)
    multi_table = mo.ui.table(
        _multi[["nertag", "support", "share_of_type", "n_roots", "pattern", "example"]],
        selection="multi",
        page_size=15,
        label=f"{len(_multi)} mitme juurega mustrit",
    )
    multi_table
    return multi_default, multi_table


@app.cell
def _(multi_default, multi_table, pattern_tabs, table):
    _selected = multi_table.value
    pattern_tabs(
        table.merge(_selected[["nertag", "pattern"]])
        if len(_selected)
        else multi_default
    )
    return


if __name__ == "__main__":
    app.run()
