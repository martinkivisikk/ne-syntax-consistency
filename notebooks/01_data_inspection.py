import marimo

__generated_with = "0.25.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from ne_syntax_consistency.db import get_storage

    return get_storage, mo


@app.cell
def _(get_storage):
    # Ühenduse andmed loetakse projekti juurkaustas olevast .env failist
    storage = get_storage()
    return (storage,)


@app.cell
def _(storage):
    # Info olemasolevate kollektsioonide kohta
    storage.collections
    return


@app.cell
def _(storage):
    # Võtame ette lausete testkollektsiooni
    collection = storage['koondkorpus_sentences_test_5000_sg_thread']
    collection
    return (collection,)


@app.cell
def _(collection):
    # kollektsiooni kihid
    collection.layers
    return


@app.cell
def _(collection, mo):
    # Näide: Üle kollektsiooni itereerimine (10 esimest lauset koos nimeüksuste ja süntaksiga):
    for text_id, text_obj in collection.select(layers=['v171_named_entities', 'v172_stanza_syntax', 'morph_analysis']).head(10):
        if len(text_obj['v171_named_entities']) > 0:
            # Kuvame ainult laused, kus leidus nimeüksuseid
            mo.output.append(mo.md(f"**{text_id}** {text_obj.text}"))
            mo.output.append(text_obj['v171_named_entities'])
            mo.output.append(text_obj['v172_stanza_syntax'])
    return


@app.cell
def _(mo):
    # marimo käivitab rakud sõltuvuste järjekorras, seega sulgemine käib nupuga
    close_button = mo.ui.run_button(label="Sulge ühendus")
    close_button
    return (close_button,)


@app.cell
def _(close_button, mo, storage):
    mo.stop(not close_button.value)
    storage.close()
    return


if __name__ == "__main__":
    app.run()
