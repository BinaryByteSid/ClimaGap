"""Shared paths, file helpers and the lexicon-based entity tagger used by every stage."""
import json
import re
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
EVAL = ROOT / "eval"

PASSAGES = PROCESSED / "passages.jsonl"
EXTRACTIONS = PROCESSED / "extractions.jsonl"
GRAPH_JSON = PROCESSED / "graph.json"
GRAPH_TTL = PROCESSED / "graph.ttl"
GAPS_CSV = PROCESSED / "gaps.csv"

ENTITY_TYPES = ("Hazard", "Sector", "Region")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def node_id(kind, name):
    return f"{kind}:{name}"


def node_kind(nid):
    return nid.split(":", 1)[0]


def node_name(nid):
    return nid.split(":", 1)[1]


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


@lru_cache(maxsize=1)
def load_lexicon():
    return json.loads((DATA / "lexicon.json").read_text(encoding="utf-8"))


def entity_names(kind):
    return list(load_lexicon()[kind])


def sdg_targets():
    return load_lexicon()["SDG"]


@lru_cache(maxsize=1)
def build_nlp():
    """A blank spaCy pipeline: sentencizer + EntityRuler built from lexicon.json (no model download)."""
    import spacy

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns = []
    for label in ENTITY_TYPES:
        for canonical, aliases in load_lexicon()[label].items():
            for alias in {canonical.lower(), *aliases}:
                patterns.append({"label": label, "pattern": alias, "id": canonical})
    ruler.add_patterns(patterns)
    return nlp


def tag_entities(text):
    """Canonical entity node ids (e.g. 'Hazard:Heatwave') mentioned in text."""
    return {node_id(e.label_, e.ent_id_) for e in build_nlp()(text).ents}
