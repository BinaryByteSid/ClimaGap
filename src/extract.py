"""Stage 2: knowledge extraction.

For every passage: tag Hazard / Sector / Region entities (lexicon NER), then find
policy measures with trigger patterns such as "will establish <object>" and link
each measure to the hazards and sectors in its sentence (falling back to the
passage when the sentence names none), plus the SDG 13 targets it serves.
"""
import re

from common import (EXTRACTIONS, PASSAGES, build_nlp, node_id, node_kind, read_jsonl,
                    sdg_targets, slug, write_jsonl)

VERBS = ["implement", "establish", "promote", "develop", "strengthen", "construct", "build",
         "introduce", "expand", "scale up", "set up", "launch", "prepare", "provide", "install",
         "create", "restore", "adopt", "upgrade", "deploy", "enforce", "undertake", "lay",
         "distribute", "mainstream"]
VERB_ALT = "|".join(v.replace(" ", r"\s+") for v in VERBS)
TRIGGER = re.compile(rf"\b(?:will|shall|to|would|should|must|and)\s+(?:also\s+|further\s+)?(?:{VERB_ALT})\b", re.I)
STOP = re.compile(
    rf"\s(?:to|for|in order|across|in|that|which|so|under|with|by|during|along|on|at|from|against|"
    rf"through|within|among|about|into|over|where|while|before|after|covering|following|"
    rf"is|are|was|were|has|have|can|may|shall|will|should|would|"
    rf"and\s+(?:{VERB_ALT}))\b|[,;:()●•▪■◦]|\s(?:ST|MT|LT)\b",
    re.I,
)
# Real plans also list actions as table rows / headings: "Promote drip irrigation",
# "Strengthening of early warning systems", "Construction of check dams". Case-sensitive on purpose.
NOMINALS = ["Development", "Construction", "Establishment", "Promotion", "Implementation", "Expansion",
            "Installation", "Restoration", "Creation", "Upgradation", "Preparation", "Strengthening",
            "Introduction", "Adoption", "Deployment", "Provision", "Setting up", "Scaling up", "Enforcement",
            "Mainstreaming", "Distribution", "Building"]
GERUNDS = ["Promoting", "Developing", "Strengthening", "Establishing", "Constructing", "Implementing",
           "Expanding", "Creating", "Restoring", "Upgrading", "Preparing", "Introducing", "Installing",
           "Adopting", "Providing", "Deploying", "Building"]
IMPERATIVES = [v[0].upper() + v[1:] for v in VERBS]
NOMINALS += ["Breeding", "Identification", "Monitoring", "Rehabilitation", "Conservation", "Augmentation"]
CAP_TRIGGER = re.compile(
    r"(?:(?:" + "|".join(n.replace(" ", r"\s+") for n in NOMINALS) + r")\s+of"
    r"|(?:" + "|".join(GERUNDS + [v.replace(" ", r"\s+") for v in IMPERATIVES]) + r"))\b"
)
# Descriptive phrasing inside prose: "is in the process of developing heat action plans",
# "and development of nutritional strategies". Lowercase forms, so only after a linking word.
LOWER_TRIGGER = re.compile(
    r"\b(?:of|in|and|by|for|through|on)\s+(?:"
    + "|".join(g.lower() for g in GERUNDS)
    + r"|(?:" + "|".join(n.lower().replace(" ", r"\s+") for n in NOMINALS) + r")\s+of)\b"
)
LEADING = re.compile(r"^(?:the|a|an|its|their|new|more|additional|all|of|and|or|also|various|appropriate)\s+", re.I)
PRONOUN_START = re.compile(r"^(?:it|they|them|this|these|those|such|same|any|both|other|which|that|we|our)\b", re.I)
TRAILING_JUNK = re.compile(r"[\s\d.,;:()\-–•●▪|/]+$")
GENERIC = {"capacity", "state", "plan", "action plan", "activities", "measures", "strategies", "actions"}
GENERIC_WORDS = {"approach", "approaches", "scheme", "schemes", "programme", "programmes", "program", "programs",
                 "initiative", "initiatives", "matrix", "framework", "mechanism", "mechanisms", "multi", "sector",
                 "adaptation", "strategy", "strategies", "activity", "activities", "measure", "measures", "/"}
DANGLING = re.compile(r"\s+(?:of|the|and|for|to|in|a|an|or|with)$", re.I)
MAX_WORDS = 10
LONG_SENTENCE = 400   # flattened PDF tables produce huge "sentences"; don't attribute across them


def measure_objects(sentence):
    """(start, end, object) for each trigger, e.g. 'will expand micro irrigation for ...' -> 'micro irrigation'.

    start/end delimit the measure's clause (trigger up to the next trigger), used to scope its entities.
    """
    triggers = sorted({(m.start(), m.end()) for rx in (TRIGGER, CAP_TRIGGER, LOWER_TRIGGER)
                       for m in rx.finditer(sentence)})
    triggers = [t for i, t in enumerate(triggers) if i == 0 or t[0] >= triggers[i - 1][1]]   # drop overlaps
    found = []
    for i, (start, end) in enumerate(triggers):
        rest = sentence[end:]
        cut = STOP.search(rest)
        obj = (rest[:cut.start()] if cut else rest).strip().rstrip(".")
        while LEADING.match(obj):
            obj = LEADING.sub("", obj, count=1)
        words = obj.split()[:MAX_WORDS]
        obj = TRAILING_JUNK.sub("", " ".join(words)).strip(" -–|•●")
        while DANGLING.search(obj):
            obj = DANGLING.sub("", obj)
        if (len(obj) < 4 or obj.lower() in GENERIC or PRONOUN_START.match(obj)
                or all(w.strip("/-").lower() in GENERIC_WORDS for w in obj.split())
                or not re.search(r"[a-z]{3}", obj)):
            continue
        clause_end = triggers[i + 1][0] if i + 1 < len(triggers) else len(sentence)
        found.append((start, clause_end, obj))
    return found


def sdgs_for(sentence, hazards):
    text = sentence.lower()
    targets = [t for t, spec in sdg_targets().items() if any(k in text for k in spec["keywords"])]
    if hazards and "13.1" not in targets:
        targets.append("13.1")
    return sorted(targets) or ["13.2"]


def extract_passage(passage, nlp):
    doc = nlp(passage["text"])
    ents = {node_id(e.label_, e.ent_id_) for e in doc.ents}
    by_kind = lambda ids, kind: sorted(i for i in ids if node_kind(i) == kind)
    measures = []
    for sent in doc.sents:
        sent_ents = {node_id(e.label_, e.ent_id_) for e in sent.ents}
        long_sentence = len(sent.text) > LONG_SENTENCE
        for start, end, name in measure_objects(sent.text):
            # Entities in the measure's own clause first, then the sentence, then the passage.
            clause = {node_id(e.label_, e.ent_id_) for e in sent.ents
                      if start <= e.start_char - sent.start_char < end}
            scopes = [("clause", clause)] + ([] if long_sentence else [("sentence", sent_ents)]) + [("passage", ents)]
            hazards = sectors = []
            level = "passage"
            for level, scope in scopes:
                hazards, sectors = by_kind(scope, "Hazard"), by_kind(scope, "Sector")
                if hazards or sectors:
                    break
            if sectors and not hazards:   # the hazard is usually named in the surrounding text or heading
                hazards = by_kind(sent_ents, "Hazard") or by_kind(ents, "Hazard")
            if not hazards and not sectors:
                continue
            context = (sent.text[start:end] if long_sentence else sent.text).strip()
            region = passage["region"] or "National"
            measures.append({
                "id": f"Measure:{region}:{slug(name)}",
                "name": name[0].upper() + name[1:],
                "region": passage["region"],
                "sentence": context,
                "hazards": hazards,
                "sectors": sectors,
                "sdgs": sdgs_for(context, hazards),
                "level": level,
            })
    return {"passage_id": passage["id"], "entities": sorted(ents), "measures": measures}


def main():
    passages = read_jsonl(PASSAGES)
    nlp = build_nlp()
    rows = [extract_passage(p, nlp) for p in passages]
    write_jsonl(EXTRACTIONS, rows)
    n_measures = sum(len(r["measures"]) for r in rows)
    n_ents = sum(len(r["entities"]) for r in rows)
    print(f"Extracted {n_ents} entity mentions and {n_measures} policy measures "
          f"from {len(rows)} passages -> {EXTRACTIONS.name}")


if __name__ == "__main__":
    main()
