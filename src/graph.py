"""Stage 3: assemble the knowledge graph, save it as JSON and export RDF/Turtle for SPARQL.

Run with --sparql to print the demo SPARQL queries' results.
"""
import csv
import json
import re
import sys
from collections import Counter

import networkx as nx

from common import (DATA, ENTITY_TYPES, EXTRACTIONS, GRAPH_JSON, GRAPH_TTL, PASSAGES,
                    load_lexicon, node_id, read_jsonl, sdg_targets)

KG_NS = "http://climagap.local/kg#"
RDF_CLASS = {"Region": "Region", "Hazard": "Hazard", "Sector": "Sector", "Measure": "PolicyMeasure",
             "Doc": "Document", "Passage": "Passage", "SDG": "SDGTarget"}
RDF_PROP = {"EXPOSED_TO": "exposedTo", "AFFECTS": "affects", "ADDRESSES": "addresses",
            "PROPOSED_IN": "proposedIn", "MENTIONED_IN": "mentionedIn", "CONTRIBUTES_TO": "contributesTo",
            "HAS_PLAN": "hasPlan", "PART_OF": "partOf", "IN_REGION": "inRegion"}


def read_csv(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_edge(G, s, o, rel, evidence=None, **attrs):
    if G.has_edge(s, o):
        if evidence and evidence not in G[s][o]["evidence"]:
            G[s][o]["evidence"].append(evidence)
        return
    G.add_edge(s, o, rel=rel, evidence=[evidence] if evidence else [], **attrs)


def build_graph():
    G = nx.DiGraph()
    lex = load_lexicon()
    for kind in ENTITY_TYPES:
        for name in lex[kind]:
            G.add_node(node_id(kind, name), type=kind, label=name)
    for target, spec in sdg_targets().items():
        G.add_node(node_id("SDG", target), type="SDG", label=f"SDG {target}", description=spec["label"])

    for row in read_csv("hazards.csv"):
        add_edge(G, node_id("Region", row["region"]), node_id("Hazard", row["hazard"]),
                 "EXPOSED_TO", severity=row["severity"], basis=row["basis"])
    for row in read_csv("impacts.csv"):
        add_edge(G, node_id("Hazard", row["hazard"]), node_id("Sector", row["sector"]), "AFFECTS")

    passages = {p["id"]: p for p in read_jsonl(PASSAGES)}
    for p in passages.values():
        doc = node_id("Doc", p["doc_id"])
        if doc not in G:
            G.add_node(doc, type="Doc", label=p["title"], region=p["region"], kind=p["kind"])
            if p["region"]:
                add_edge(G, node_id("Region", p["region"]), doc, "HAS_PLAN")
        pid = node_id("Passage", p["id"])
        G.add_node(pid, type="Passage", label=p["id"], text=p["text"], page=p["page"],
                   doc=p["doc_id"], region=p["region"])
        add_edge(G, pid, doc, "PART_OF")

    for ex in read_jsonl(EXTRACTIONS):
        p = passages[ex["passage_id"]]
        pid = node_id("Passage", p["id"])
        for ent in ex["entities"]:
            add_edge(G, ent, pid, "MENTIONED_IN")
        for m in ex["measures"]:
            mid = m["id"]
            if mid not in G:
                G.add_node(mid, type="Measure", label=m["name"], region=m["region"], doc=p["doc_id"])
            add_edge(G, mid, pid, "MENTIONED_IN")
            add_edge(G, mid, node_id("Doc", p["doc_id"]), "PROPOSED_IN")
            if m["region"]:
                add_edge(G, mid, node_id("Region", m["region"]), "IN_REGION")
            for target in m["hazards"] + m["sectors"]:
                add_edge(G, mid, target, "ADDRESSES", evidence=p["id"], level=m["level"])
                if m["level"] != "passage":   # any direct mention upgrades an inferred link
                    G[mid][target]["level"] = m["level"]
            for sdg in m["sdgs"]:
                add_edge(G, mid, node_id("SDG", sdg), "CONTRIBUTES_TO")
    return G


def save_graph(G, path=GRAPH_JSON):
    data = {"nodes": [{"id": n, **a} for n, a in G.nodes(data=True)],
            "edges": [{"source": s, "target": t, **a} for s, t, a in G.edges(data=True)]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_graph(path=GRAPH_JSON):
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for n in data["nodes"]:
        G.add_node(n.pop("id"), **n)
    for e in data["edges"]:
        G.add_edge(e.pop("source"), e.pop("target"), **e)
    return G


def nodes_of(G, kind):
    return [n for n, a in G.nodes(data=True) if a["type"] == kind]


def out_by_rel(G, node, rel):
    return [t for t in G.successors(node) if G[node][t]["rel"] == rel]


def in_by_rel(G, node, rel):
    return [s for s in G.predecessors(node) if G[s][node]["rel"] == rel]


# ---------------------------------------------------------------- RDF / SPARQL

def uri(nid):
    from rdflib import URIRef
    return URIRef(KG_NS + re.sub(r"[^A-Za-z0-9]+", "_", nid).strip("_"))


def to_rdf(G):
    from rdflib import Graph, Literal, Namespace
    from rdflib.namespace import RDF, RDFS

    KG = Namespace(KG_NS)
    g = Graph()
    g.bind("kg", KG)
    for n, a in G.nodes(data=True):
        g.add((uri(n), RDF.type, KG[RDF_CLASS[a["type"]]]))
        g.add((uri(n), RDFS.label, Literal(a["label"])))
        if a["type"] == "Passage":
            g.add((uri(n), KG.text, Literal(a["text"])))
            g.add((uri(n), KG.page, Literal(a["page"])))
    for s, t, a in G.edges(data=True):
        g.add((uri(s), KG[RDF_PROP[a["rel"]]], uri(t)))
        if a["rel"] == "ADDRESSES" and a.get("level") != "passage":   # named in the measure's own clause
            g.add((uri(s), KG.directlyAddresses, uri(t)))
        if a["rel"] == "EXPOSED_TO":   # reify exposure so severity is queryable
            exp = uri(f"Exposure:{s}:{t}")
            g.add((exp, RDF.type, KG.Exposure))
            g.add((exp, KG.region, uri(s)))
            g.add((exp, KG.hazard, uri(t)))
            g.add((exp, KG.severity, Literal(a["severity"])))
    return g


PREFIX = f"PREFIX kg: <{KG_NS}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
SPARQL_EXAMPLES = {
    "Measures addressing heatwaves in Gujarat": PREFIX + """
SELECT ?measure WHERE {
  ?m a kg:PolicyMeasure ; kg:inRegion kg:Region_Gujarat ; kg:addresses kg:Hazard_Heatwave ;
     rdfs:label ?measure .
} ORDER BY ?measure""",
    "Hazards each state faces with NO direct measure in its plan (gaps)": PREFIX + """
SELECT ?region ?hazard ?severity WHERE {
  ?e a kg:Exposure ; kg:region ?r ; kg:hazard ?h ; kg:severity ?severity .
  ?r rdfs:label ?region . ?h rdfs:label ?hazard .
  FILTER NOT EXISTS { ?m a kg:PolicyMeasure ; kg:inRegion ?r ; kg:directlyAddresses ?h . }
} ORDER BY ?region ?hazard""",
    "Sectors at risk in Odisha (region -> hazard -> sector)": PREFIX + """
SELECT DISTINCT ?hazard ?sector WHERE {
  kg:Region_Odisha kg:exposedTo ?h . ?h kg:affects ?s .
  ?h rdfs:label ?hazard . ?s rdfs:label ?sector .
} ORDER BY ?hazard ?sector""",
    "Measures contributing to SDG 13.3 (awareness & capacity)": PREFIX + """
SELECT ?region ?measure WHERE {
  ?m a kg:PolicyMeasure ; kg:contributesTo kg:SDG_13_3 ; kg:inRegion ?r ; rdfs:label ?measure .
  ?r rdfs:label ?region .
} ORDER BY ?region""",
    "Measures per state (count)": PREFIX + """
SELECT ?region (COUNT(?m) AS ?measures) WHERE {
  ?m a kg:PolicyMeasure ; kg:inRegion ?r . ?r rdfs:label ?region .
} GROUP BY ?region ORDER BY DESC(?measures)""",
}


def run_sparql(rdf_graph, query):
    result = rdf_graph.query(query)
    cols = [str(v) for v in result.vars]
    return cols, [[str(x) for x in row] for row in result]


def main():
    G = build_graph()
    save_graph(G)
    rdf = to_rdf(G)
    rdf.serialize(GRAPH_TTL, format="turtle")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(rdf)} RDF triples")
    print("  nodes:", dict(Counter(a["type"] for _, a in G.nodes(data=True))))
    print("  edges:", dict(Counter(a["rel"] for _, _, a in G.edges(data=True))))
    print(f"Saved {GRAPH_JSON.name} and {GRAPH_TTL.name}")
    if "--sparql" in sys.argv:
        for title, q in SPARQL_EXAMPLES.items():
            cols, rows = run_sparql(rdf, q)
            print(f"\n# {title}\n  {' | '.join(cols)}")
            for r in rows:
                print("  " + " | ".join(r))


if __name__ == "__main__":
    main()
