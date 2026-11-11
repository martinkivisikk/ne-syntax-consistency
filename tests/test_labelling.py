import pandas as pd

from ne_syntax_consistency.labelling import (
    boundary_suggestion,
    first_unlabelled,
    join_note,
    load_labels,
    problem,
    save_labels,
    split_note,
)


def test_note_check_prefix_roundtrip():
    assert join_note(True, " metonymy ") == "CHECK: metonymy"
    assert join_note(False, "x") == "x"
    assert split_note("CHECK: metonymy") == (True, "metonymy")
    assert split_note(join_note(True, "")) == (True, "")
    assert split_note("plain") == (False, "plain")


def test_problem_follows_guidelines():
    assert problem(None, "", "", "PER") == "choose an ne value"
    assert problem("ok", "", "", "PER") is None
    assert problem("ok", "LOC", "", "PER") is not None  # fix only for type/boundary
    assert problem("type", "LOC", "", "PER") is None
    assert problem("type", "PER", "", "PER") is not None  # same as the tag
    assert problem("type", "MISC", "", "PER") is not None  # outside the tagset
    assert problem("boundary", "[[EMÜ]] ORG", "", "PER") is None
    assert problem("boundary", "EMÜ", "", "PER") is not None
    assert problem("not_ne", "", "", "PER") is not None  # needs a reason
    assert problem("not_ne", "", "outside tagset: product", "PER") is None
    assert problem("unclear", "", "CHECK: ", "PER") is not None


def test_boundary_suggestion_strips_edge_punctuation():
    assert boundary_suggestion("/ EMÜ", "PER") == "[[EMÜ]] PER"
    assert boundary_suggestion("' Aspromonte", "PER") == "[[Aspromonte]] PER"
    assert boundary_suggestion("O ' Neal", "PER") == "[[O ' Neal]] PER"


def test_save_and_load_keep_empty_labels(tmp_path):
    path = tmp_path / "ex.csv"
    df = pd.DataFrame(
        {"text_id": [1, 2], "ne": ["ok", ""], "ne_fix": ["", ""], "note": ["", ""]}
    )
    save_labels(df, path)
    back = load_labels(path)
    assert back["ne"].tolist() == ["ok", ""]
    assert back["note"].tolist() == ["", ""]
    assert first_unlabelled(back) == 1
    assert not (tmp_path / "ex.csv.tmp").exists()
