"""Loaders shared by the dev/subdiv_0*.py steps. Nothing here prints data.

Deed key = (county lower case, frozenset of image_ids), the key used by dev/infer_try08.py.
"""
import ast
import re
from collections import defaultdict

EMPTY = {"", "[]", "none", "nan", "null"}
ID_COLUMNS = ["doc_num", "zn_subj_id", "image_ids", "cnty_name"]
GOLD_COLUMNS = ID_COLUMNS + ["add_cov", "add_mod", "block_cov", "lot_cov", "block_mod", "lot_mod"]


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip()
    return None if s.lower() in EMPTY else s


def parse_list(value):
    if hasattr(value, "tolist") and not isinstance(value, str):
        value = value.tolist()
    if value is None:
        return []
    if isinstance(value, str):
        if value.strip().lower() in EMPTY:
            return []
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [value.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if clean(v)]
    return [str(value).strip()]


def parse_ids(value):
    """image_ids as a list: "('a', 'b')" (our splits) or "a,b" (mp-us-racial-covenants)."""
    if isinstance(value, str) and not value.strip().startswith(("(", "[")):
        return [x.strip() for x in value.split(",") if x.strip()]
    return parse_list(value)


def norm_doc(v):
    """doc_num as a comparable string: strip, upper, drop a float '.0' artefact and leading zeros."""
    s = clean(v)
    if s is None:
        return None
    s = re.sub(r"\.0$", "", s.upper())
    return s.lstrip("0") or "0"


def norm_id(v):
    s = clean(v)
    return None if s is None else re.sub(r"\.0$", "", s)


def county_key(v):
    s = (clean(v) or "").lower()
    return re.sub(r"\s+county$", "", s)


def deed_key(row):
    return (county_key(row.get("cnty_name")), frozenset(parse_ids(row.get("image_ids"))))


def read_split(path, columns=GOLD_COLUMNS):
    """Split GeoJSON -> list of dict rows with only `columns` (no geometry, no OCR, no cov_text)."""
    import pyogrio
    df = pyogrio.read_dataframe(path, columns=columns, read_geometry=False)
    return df.to_dict("records")


def exclusion_ids(rows):
    """Identifiers of a split, for removing its deeds from other sources (no labels, no text)."""
    out = {"doc_num": set(), "zn_subj_id": set(), "image_id": set(), "deed_key": set()}
    for r in rows:
        c = county_key(r.get("cnty_name"))
        if norm_doc(r.get("doc_num")):
            out["doc_num"].add((c, norm_doc(r.get("doc_num"))))
        if norm_id(r.get("zn_subj_id")):
            out["zn_subj_id"].add(norm_id(r.get("zn_subj_id")))
        for i in parse_ids(r.get("image_ids")):
            out["image_id"].add(i)
        out["deed_key"].add(deed_key(r))
    return out


def deeds_from_rows(rows):
    """Split rows (one per parcel) -> {deed_key: {county, doc_nums, add_cov, add_mod, block_cov, lot_cov,
    block_mod, lot_mod}} with list values in first-appearance order (duplicates dropped)."""
    deeds = {}
    fields = ("add_cov", "add_mod", "block_cov", "lot_cov", "block_mod", "lot_mod")
    for r in rows:
        k = deed_key(r)
        d = deeds.setdefault(k, {"county": k[0], "doc_nums": [], **{f: [] for f in fields}})
        dn = norm_doc(r.get("doc_num"))
        if dn and dn not in d["doc_nums"]:
            d["doc_nums"].append(dn)
        for f in fields:
            v = clean(r.get(f))
            if v and v not in d[f]:
                d[f].append(v)
    return deeds


def read_predictions(path):
    """dev/infer_try08.py output -> {deed_key: {"SUBDIVISION_p": [...], "n_parts": int}}."""
    import pyogrio
    df = pyogrio.read_dataframe(path, columns=["image_ids", "cnty_name", "SUBDIVISION_p", "n_parts"],
                                read_geometry=False)
    out = {}
    for r in df.to_dict("records"):
        out[deed_key(r)] = {"SUBDIVISION_p": parse_list(r.get("SUBDIVISION_p")),
                            "n_parts": int(r.get("n_parts") or 0)}
    return out


def group_by(items, key):
    out = defaultdict(list)
    for x in items:
        out[key(x)].append(x)
    return out
