from ne_syntax_consistency.data import ud_ne_spans, ud_ne_tag, ud_records

CONLLU = """\
# sent_id = doc1_1
# text = Jürgen Zopp võitis Tallinnas USA-d .
1\tJürgen\tJürgen\tPROPN\tS\tCase=Nom|Number=Sing\t3\tnsubj\t_\tNE=B-Per
2\tZopp\tZopp\tPROPN\tS\tCase=Nom|Number=Sing\t1\tflat\t_\tNE=I-Per
3\tvõitis\tvõitma\tVERB\tV\tMood=Ind\t0\troot\t_\t_
4\tTallinnas\tTallinn\tPROPN\tS\tCase=Ine|Number=Sing\t3\tobl\t_\tNE=B_Gep
5\tUSA-d\tUSA\tPROPN\tY\tAbbr=Yes|Case=Par|Number=Sing\t3\tobj\t_\tNE=BäOrg|SpaceAfter=No
6\t.\t.\tPUNCT\tZ\t_\t3\tpunct\t_\t_

"""


def test_ud_ne_tag_repairs_broken_tags():
    assert ud_ne_tag("B-Per") == ("B", "PER")
    assert ud_ne_tag("BäOrg") == ("B", "ORG")
    assert ud_ne_tag("B_Gep") == ("B", "GEP")
    assert ud_ne_tag("i-Prod") == ("I", "PROD")
    assert ud_ne_tag("B-OrgSpaceAfter=No") == ("B", "ORG")
    assert ud_ne_tag("Org") == ("I", "ORG")
    assert ud_ne_tag(None) is None


def test_ud_ne_spans():
    tags = [("B", "PER"), ("I", "PER"), None, ("I", "LOC"), ("B", "LOC"), ("I", "ORG")]
    assert ud_ne_spans(tags) == [
        ("PER", [0, 1]),
        ("LOC", [3]),
        ("LOC", [4]),
        ("ORG", [5]),
    ]


def test_ud_records(tmp_path):
    path = tmp_path / "x.conllu"
    path.write_text(CONLLU, encoding="utf-8")
    (r,) = list(ud_records(path))
    assert r["sent_id"] == "doc1_1"
    assert r["pos"] == ["H", "H", "V", "H", "Y", "Z"]  # PROPN -> H, but PROPN/Y stays Y
    assert r["form"] == ["sg n", "sg n", "", "sg in", "sg p", ""]
    assert r["head"] == [3, 1, 0, 3, 3, 3]
    assert r["ne_tag"] == ["PER", "GEP", "ORG"]
    assert r["ne_words"] == [[0, 1], [3], [4]]
    assert len(r["vm_pos"]) == len(r["words"])
