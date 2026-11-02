# ClimaGap: a Knowledge Graph that finds gaps in climate adaptation policy

**KGIRS microproject · SDG 13 (Climate Action), target 13.2: integrate climate measures into policies and planning**

> Which climate hazards does a state actually face, and which of them does its own climate action plan *fail* to address?

ClimaGap reads five official State Action Plans on Climate Change (SAPCCs), builds a knowledge graph of
**Region → Hazard → Sector ← Policy Measure**, answers questions with hybrid retrieval, and flags
**policy gaps**: hazards a state is exposed to that its plan never explicitly acts on, or acts on for
only some of the affected sectors. Every gap links back to the plan passages that support it.

## Corpus

| State | Document | Pages | Passages |
|---|---|---|---|
| Gujarat | Second SAPCC (**draft, marked "not to be cited or quoted"**) | 311 | 650 |
| Maharashtra | SAPCC 2020–2030 | 340 | 1,243 |
| Odisha | Climate Change Action Plan 2021–2030 | 230 | 550 |
| Kerala | SAPCC 2023–2030 | 187 | 730 |
| Rajasthan | SAPCC 2014 (the portal has no revised edition) | 219 | 661 |

All five come from the MoEFCC NAPCC portal (`napccindia.moef.gov.in`); URLs are in `data/sources.csv`.
`data/hazards.csv` (region → hazard → severity) is **indicative**: verify it against NDMA hazard
profiles before quoting severities. The earlier illustrative sample texts are kept in `data/samples/`
with their query set `eval/queries_sample.json`.

## Architecture

```
data/raw/*.pdf ──► ingest.py (pypdf, header/footer removal, page-text cache) ──► passages.jsonl
                                        │
lexicon.json ──► extract.py (spaCy EntityRuler NER + pattern-based measure extraction)
                                        │
hazards.csv, impacts.csv ──► graph.py ──► graph.json (NetworkX) + graph.ttl (RDF, SPARQL)
                                        │
                     ┌──────────────────┼───────────────────┐
              retrieve.py           gaps.py            evaluate.py
       BM25 · Dense · Hybrid    coverage matrix     P@5 · R@10 · MRR · nDCG
       + graph re-rank          + ranked gaps
                     └──────────────► app.py (Streamlit) ◄──┘
```

**Graph:** 5,824 nodes (3,834 passages, ~2,000 policy measures, 5 states, 6 hazards, 7 sectors, 3 SDG targets), about 25,800 edges, and about 45,000 RDF triples.

### Knowledge graph schema

| Node | Example | | Edge | Meaning |
|---|---|---|---|---|
| Region | Gujarat | | `EXPOSED_TO` (severity) | Region → Hazard |
| Hazard | Heatwave | | `AFFECTS` | Hazard → Sector |
| Sector | Health | | `ADDRESSES` (evidence, level) | Measure → Hazard / Sector |
| PolicyMeasure | Cyclone shelters | | `IN_REGION`, `PROPOSED_IN` | Measure → Region / Document |
| Document, Passage | Odisha SAPCC, p.141 | | `HAS_PLAN`, `PART_OF` | Region → Doc, Passage → Doc |
| SDGTarget | 13.1 / 13.2 / 13.3 | | `MENTIONED_IN`, `CONTRIBUTES_TO` | Entity → Passage, Measure → SDG |

### How each KGIRS concept is used
- **Information extraction:** lexicon NER with synonym normalisation ("heat wave", "heatstroke", "heat action plan" → *Heatwave*). Policy measures come from three trigger families: "will establish / expand …", table-style "Strengthening of … / Promote …", and descriptive "in the process of developing …".
- **Relation extraction:** each measure is linked to the hazards and sectors named in **its own clause** (direct). It falls back to the passage (indirect) only when the clause names none, and the link level is stored on the edge.
- **Knowledge representation:** a property graph plus an RDF/Turtle export queried with SPARQL (e.g. a `FILTER NOT EXISTS` query that lists gaps).
- **Retrieval models:** BM25, dense MiniLM embeddings, min-max-normalised hybrid fusion, and a **graph re-rank** that boosts passages linked to the query's entities and their 1-hop neighbourhood.
- **Evaluation:** TREC-style pooled relevance judgments; P@5, R@10, MRR, nDCG@10.
- **Reasoning over the graph:** gap analysis, i.e. a Region → Hazard → Sector path that no direct measure of that region closes.

### Gap statuses
`coverage = 0.5 × min(1, direct measures / 5) + 0.5 × share of affected sectors addressed directly`

| Status | Meaning |
|---|---|
| Covered | 5+ direct measures, and every affected sector is addressed |
| Partial | direct measures exist, but fewer than 5 or some affected sectors are missing |
| Weak evidence | only indirect (passage-level) links, so no measure explicitly targets the hazard |
| Mentioned, no action | the plan discusses the hazard but proposes nothing linked to it |
| Not mentioned | the state is exposed, but the plan never mentions the hazard |

`priority = severity weight (high 3 / medium 2 / low 1) × (1 − coverage)`

## Run

```bash
pip install -r requirements.txt
python run_pipeline.py --eval     # ingest → extract → graph → gaps → evaluation
streamlit run app.py
```

The first ingestion reads about 1,300 PDF pages (about 4 minutes) and caches the text in
`data/processed/pdf_text/`. The first app start also embeds the corpus with `all-MiniLM-L6-v2`
(downloaded once, about 90 MB), which takes 1–3 minutes; after that each interaction takes well
under a second. If the model is unavailable, dense/hybrid fall back to BM25.

## Findings (automatic extraction — verify before quoting)

| Rank | State × hazard | Status | What the extractor found |
|---|---|---|---|
| 1 | **Odisha × sea-level rise** (high) | Weak evidence | The plan describes sea-level-rise vulnerability at length (pp. 30–34, 93–99). Its coastal actions (saline embankments, shoreline management) are filed under flood management, and none is framed as sea-level-rise adaptation. |
| 2 | Rajasthan × heatwave (high) | Partial | Only 4 direct measures (livestock heat stress, health surveillance), and nothing on energy or urban heat. Note that this is the older 2014 plan. |
| 3 | Kerala × sea-level rise (high) | Partial | Coastal protection measures exist, but no direct link to agriculture or water (saltwater intrusion into farmland and aquifers). |
| 4 | Kerala × landslide (high) | Partial | 5 direct measures (hazard zonation, land-use assessment), but no forest or slope-vegetation measure is tied to landslides. |
| — | **Heatwave × Energy** | Partial in Gujarat, Maharashtra, Rajasthan | A cross-state pattern: heat actions focus on health, and cooling demand and grid stress during heatwaves are rarely linked. Kerala's heat action plan does assign the power department duties. |

**Spot checks against the PDFs** (do the same for any gap you report):
- **Confirmed:** Odisha × sea-level rise, with the nuance described in the table.
- **Corrected during development:** Gujarat × heatwave × Health was a false gap. The plan cites Ahmedabad's Heat Action Plan as a health early-warning system and says GIDM "is in the process of developing heat action plans". This led to the descriptive-phrasing triggers and the "heat action plan" lexicon alias.
- **Extractor recall:** Rajasthan's heat actions were initially missed for the same reason. A "Partial" means *the extractor found fewer actions*, not necessarily that the plan lacks them.

## Retrieval evaluation (real corpus: 15 queries, 92 judged passages)

| Method | P@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|
| BM25 | 0.613 | 0.643 | 0.767 | 0.607 |
| Dense | 0.587 | 0.628 | 0.789 | 0.559 |
| **Hybrid** | 0.613 | **0.773** | **0.856** | **0.692** |
| Hybrid + KG | **0.680** | 0.725 | 0.850 | 0.665 |

- **Hybrid fusion** beats either model alone on every metric except P@5, the classic result.
- **The graph re-rank gives the best top-5 precision** but is not a uniform win. It helps most when the query names a specific hazard: *landslide risk in Kerala* +0.31 nDCG, *flood early warning* +0.19.
- **It hurts when the query's entities are ubiquitous:** *crop insurance … drought* −0.45. *Drought* and *Agriculture* are tagged in hundreds of passages, so the boost floods the results with generic drought text. An entity-specificity weight (IDF-style) is the obvious next step.
- The weights were **not tuned** on these judgments.

How the judgments were made: the top 5 of all four methods were pooled per query, and every pooled passage was graded 0/1/2. Queries r05 and r15 were supplemented by keyword search, because the pool held few relevant passages for them (on r05, all four methods missed real landslide passages). Judgments are in `eval/queries.json` and were made while building the project, so review them before reporting. Unjudged passages count as non-relevant, which understates recall.

## Measure extraction quality
On a random sample of 40 direct hazard-linked measures, about **50%** were a real action correctly
linked to its hazard (e.g. "Drought-tolerant crop varieties", "Cyclone shelters", "Seawall", "Heat
action plans"). The rest are table headers or vague phrases ("Agencies Outlay", "Approaches") that
come from PDF tables being flattened into text. Gap analysis uses counts of direct links, so a few
noisy measures shift coverage only slightly, but **always check the evidence passages** (the Gap Map
tab shows them) before citing a gap.

## Adding or replacing plans
1. Put the PDF in `data/raw/`, named `<Region>__<anything>.pdf`. The region prefix must match `lexicon.json`.
2. Add a row to `data/sources.csv` (doc_id, title, year, `kind=real`, url).
3. Extend `lexicon.json` and `hazards.csv` if needed, then run `python run_pipeline.py --eval`.
4. Passage ids may change if chunking changes, so re-pool and re-judge `eval/queries.json`.

## Project layout
```
data/raw/            the five SAPCC PDFs
data/samples/        illustrative sample texts from the first prototype
data/lexicon.json    entity synonyms + SDG 13 keyword rules
data/hazards.csv     region → hazard exposure (indicative severity)
data/impacts.csv     hazard → affected sector
data/sources.csv     per-file metadata and source URLs
src/                 common · ingest · extract · graph · retrieve · gaps · evaluate
eval/queries.json    real-corpus judgments;  eval/queries_sample.json  sample-corpus set
app.py               Streamlit UI: Ask · Graph · Gap Map · SPARQL
run_pipeline.py      runs every stage
```

## Limitations and next steps
- Pattern-based measure extraction (~50% precision) misses passive or unusual phrasing. A trained relation extractor, or an LLM that returns citations, would improve both precision and recall.
- PDF tables are flattened into text. A table-aware extractor (e.g. pdfplumber) would separate action, agency and budget columns.
- Coverage counts whether a sector is *addressed*, not how well: budgets, timelines and implementation status aren't modelled.
- The plans are of different ages (Rajasthan 2014 vs. Kerala 2023), and Gujarat's is a draft.
- Stretch goals: IDF-weighted graph boosting, an LLM summary per gap with citations, and comparing old vs. revised SAPCCs to see which gaps closed.
