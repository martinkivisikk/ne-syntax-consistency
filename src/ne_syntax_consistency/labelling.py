"""Interactive labelling of NE examples (see labelling_guidelines.md).

The CSV from scripts/export_examples.py is labelled row by row in a Jupyter
widget (notebooks/05_labelling.ipynb): each row shows the sentence's dependency
diagram with the span highlighted, and the ne, ne_fix and note fields. Every
saved row is written to the CSV at once (atomically), so the notebook can be
closed at any time and resumes at the first unlabelled row.
"""

import os
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

CODES = ("ok", "type", "boundary", "not_ne", "unclear")
TYPES = ("PER", "LOC", "ORG")
LABEL_COLUMNS = ("ne", "ne_fix", "note")
CHECK = "CHECK: "
SENTENCE_COLUMNS = [
    "text_id",
    "words",
    "pos",
    "form",
    "head",
    "deprel",
    "ne_tag",
    "ne_words",
]
_PUNCT = re.compile(r"^[^\w]+$")


def load_labels(path: Path) -> pd.DataFrame:
    """The examples CSV with the label columns as strings ("" = empty)."""
    df = pd.read_csv(path, dtype={c: str for c in LABEL_COLUMNS}, keep_default_na=False)
    missing = [c for c in LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} has no column(s) {missing}")
    return df


def save_labels(df: pd.DataFrame, path: Path) -> None:
    """Write the CSV via a temporary file, so a crash cannot leave half a file."""
    tmp = Path(f"{path}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def split_note(note: str) -> tuple[bool, str]:
    """("CHECK: text") -> (True, "text")."""
    if note.startswith(CHECK):
        return True, note[len(CHECK) :]
    if note.strip() == CHECK.strip():
        return True, ""
    return False, note


def join_note(check: bool, text: str) -> str:
    text = text.strip()
    return (CHECK + text).strip() if check else text


def problem(ne: str | None, ne_fix: str, note: str, nertag: str) -> str | None:
    """Why a label cannot be saved under the guidelines, or None if it can."""
    ne_fix = ne_fix.strip()
    if ne not in CODES:
        return "choose an ne value"
    if ne == "type":
        if ne_fix not in TYPES:
            return f"type: ne_fix must be one of {', '.join(TYPES)}"
        if ne_fix == nertag:
            return f"type: ne_fix is the same as the tag ({nertag})"
    elif ne == "boundary":
        if "[[" not in ne_fix or "]]" not in ne_fix:
            return "boundary: write the corrected span(s), e.g. [[EMÜ]] ORG"
    elif ne_fix:
        return f"{ne}: ne_fix must be empty"
    if ne in ("not_ne", "unclear") and not split_note(note)[1].strip():
        return f"{ne}: give a short reason in note"
    return None


def boundary_suggestion(ne_text: str, nertag: str) -> str:
    """Starting point for a boundary fix: the span without edge punctuation."""
    words = ne_text.split()
    while words and _PUNCT.match(words[0]):
        words.pop(0)
    while words and _PUNCT.match(words[-1]):
        words.pop()
    return f"[[{' '.join(words)}]] {nertag}"


def first_unlabelled(df: pd.DataFrame) -> int:
    empty = (df["ne"] == "").to_numpy().nonzero()[0]
    return int(empty[0]) if len(empty) else len(df) - 1


def load_sentences(path: Path, text_ids) -> dict:
    """Sentence records (data.SENTENCE_SCHEMA) for the given text_ids only."""
    table = pq.read_table(
        path,
        columns=SENTENCE_COLUMNS,
        filters=[("text_id", "in", sorted({int(t) for t in text_ids}))],
    )
    return {s["text_id"]: s for s in table.to_pylist()}


class Labeller:
    """Jupyter widget that goes through the rows of `df` and saves to `path`."""

    def __init__(self, df, path, sentences, window_size=5, style=None):
        import ipywidgets as w

        self.df, self.path, self.sentences = df, Path(path), sentences
        self.window_size, self.style = window_size, style
        self.i = first_unlabelled(df)

        self.progress = w.HTML()
        self.info = w.HTML()
        self.whole = w.Checkbox(description="whole sentence", indent=False)
        self.diagram = w.HTML()
        self.ne = w.ToggleButtons(
            options=CODES,
            value=None,
            tooltips=[
                "PER/LOC/ORG entity, exact span, correct type",
                "correct span, wrong type",
                "entity there, but span too long/short, merged or split",
                "no PER/LOC/ORG entity (incl. names outside the tagset)",
                "cannot decide from the sentence",
            ],
            style={"button_width": "90px"},
        )
        self.ne_fix = w.Text(description="ne_fix", layout=w.Layout(width="70%"))
        self.note = w.Text(description="note", layout=w.Layout(width="70%"))
        self.check = w.Checkbox(description="CHECK (borderline)", indent=False)
        self.message = w.HTML()
        self.prev_button = w.Button(description="◀ prev", layout=w.Layout(width="90px"))
        self.next_button = w.Button(description="save ▶ next", button_style="primary")
        self.skip_button = w.Button(description="next unlabelled")
        self.goto = w.BoundedIntText(
            min=1, max=len(df), description="row", layout=w.Layout(width="150px")
        )
        self.goto_button = w.Button(description="go", layout=w.Layout(width="50px"))

        self.ne.observe(self._on_ne, names="value")
        self.whole.observe(lambda _: self._draw(), names="value")
        self.prev_button.on_click(lambda _: self._move(self.i - 1))
        self.next_button.on_click(lambda _: self._move(self.i + 1))
        self.skip_button.on_click(lambda _: self._move(self._next_unlabelled()))
        self.goto_button.on_click(lambda _: self._move(self.goto.value - 1))

        self.widget = w.VBox(
            [
                self.progress,
                self.info,
                self.whole,
                self.diagram,
                self.ne,
                self.ne_fix,
                w.HBox([self.note, self.check]),
                self.message,
                w.HBox(
                    [
                        self.prev_button,
                        self.next_button,
                        self.skip_button,
                        self.goto,
                        self.goto_button,
                    ]
                ),
            ]
        )
        self._show()

    # navigation ------------------------------------------------------------

    def _next_unlabelled(self) -> int:
        after = [j for j in range(self.i + 1, len(self.df)) if not self.df.at[j, "ne"]]
        before = [j for j in range(self.i) if not self.df.at[j, "ne"]]
        return (after or before or [self.i])[0]

    def _move(self, j: int) -> None:
        """Save the current row (if a value is chosen), then show row j."""
        if self.ne.value is not None and not self._save():
            return
        self.i = max(0, min(j, len(self.df) - 1))
        self._show()

    def _save(self) -> bool:
        row = self.df.loc[self.i]
        note = join_note(self.check.value, self.note.value)
        why = problem(self.ne.value, self.ne_fix.value, note, row["nertag"])
        if why:
            self.message.value = f'<b style="color:#c0392b">Not saved: {why}</b>'
            return False
        new = (self.ne.value, self.ne_fix.value.strip(), note)
        if tuple(row[c] for c in LABEL_COLUMNS) != new:
            for c, v in zip(LABEL_COLUMNS, new):
                self.df.at[self.i, c] = v
            save_labels(self.df, self.path)
        return True

    # display ---------------------------------------------------------------

    def _on_ne(self, change) -> None:
        row = self.df.loc[self.i]
        if change["new"] == "boundary" and not self.ne_fix.value:
            self.ne_fix.value = boundary_suggestion(row["ne_text"], row["nertag"])
        elif change["new"] in ("ok", "not_ne", "unclear") and change["old"] in (
            "type",
            "boundary",
        ):
            self.ne_fix.value = ""
        placeholder = {
            "type": "PER / LOC / ORG",
            "boundary": "[[corrected span]] TYPE  (several: ' / ')",
        }
        self.ne_fix.placeholder = placeholder.get(change["new"], "(empty)")
        self.note.placeholder = (
            "outside tagset: product / event / ... or other reason"
            if change["new"] == "not_ne"
            else "short reason"
        )

    def _draw(self) -> None:
        from ne_syntax_consistency.viz import record_svg

        row = self.df.loc[self.i]
        s = self.sentences.get(int(row["text_id"]))
        if s is None:
            self.diagram.value = "<i>sentence not found in the sentence data</i>"
            return
        span = s["ne_words"][int(row["ne_index"])]
        size = None if self.whole.value else self.window_size
        svg = record_svg(s, span, size, self.style)
        self.diagram.value = f'<div style="overflow-x:auto">{svg}</div>'

    def _show(self) -> None:
        import html

        row = self.df.loc[self.i]
        done = int((self.df["ne"] != "").sum())
        self.progress.value = (
            f"<b>row {self.i + 1} / {len(self.df)}</b> · labelled {done} · "
            f"{row['nertag']} pattern #{row['rank']} <code>"
            f"{html.escape(str(row['pattern']))}</code> "
            f"(support {row['support']})"
        )
        sentence = html.escape(str(row["sentence"]))
        sentence = re.sub(
            r"\[\[(.+?)\]\]",
            r'<mark style="background:#ffe08a">[[\1]]</mark>',
            sentence,
        )
        self.info.value = (
            f"<div style='font-size:1.1em;margin:6px 0'>"
            f"<b>{html.escape(str(row['ne_text']))}</b> = {row['nertag']} "
            f"<span style='color:gray'>({row['ne_text_count']}× in this pattern, "
            f"text_id {row['text_id']})</span></div>"
            f"<div style='margin:6px 0'>{sentence}</div>"
            f"<div style='color:gray'>other NEs: "
            f"{html.escape(str(row['other_nes'])) or '–'} · root: "
            f"{html.escape(str(row['root_heads']))}</div>"
        )
        self._draw()
        check, note = split_note(row["note"])
        self.ne.value = row["ne"] or None
        self.ne_fix.value = row["ne_fix"]
        self.note.value = note
        self.check.value = check
        self.goto.value = self.i + 1
        self.message.value = (
            "<b style='color:#27ae60'>all rows labelled</b>"
            if done == len(self.df)
            else ""
        )
