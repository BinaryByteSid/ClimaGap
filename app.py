"""ClimaGap: ask questions over state climate plans, see which hazards they leave uncovered, and explore the map.

    streamlit run app.py
"""
import re
import sys
from math import cos, pi, sin
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from common import EVAL, GRAPH_JSON, GRAPH_TTL, node_id, node_kind, node_name
from gaps import addresses, analyze, coverage_matrix
from graph import SPARQL_EXAMPLES, in_by_rel, load_graph, nodes_of, out_by_rel, run_sparql
from retrieve import METHODS, Retriever

st.set_page_config(page_title="ClimaGap", page_icon="🌍", layout="wide")

STATUS_COLOR = {"Covered": "#2e7d32", "Partial": "#f9a825",
                "Weak evidence": "#e65100", "Mentioned, no action": "#ef6c00", "Not mentioned": "#c62828"}
STATUS_MEANING = {"Covered": "has a good plan", "Partial": "has half a plan",
                  "Weak evidence": "has no clear plan", "Mentioned, no action": "is talked about, but has no plan",
                  "Not mentioned": "is never mentioned"}
KIND_COLOR = {"Region": "#1565c0", "Hazard": "#c62828", "Sector": "#6d4c41",
              "Measure": "#2e7d32", "SDG": "#00838f"}
METHOD_LABEL = {"bm25": "Keywords only (BM25)", "dense": "Meaning only (embeddings)", "hybrid": "Keywords + meaning",
                "hybrid_kg": "Keywords + meaning + knowledge graph (recommended)"}
EXAMPLES = ["Heat action plan for Ahmedabad", "Cyclone shelters in coastal Odisha", "Urban flooding in Mumbai",
            "Landslide risk in Kerala", "Drought-tolerant crops in Rajasthan", "Mangroves to protect the coast"]
MEASURES_PER_HAZARD = 3

if not GRAPH_JSON.exists():
    st.error("No knowledge graph yet. Run `python run_pipeline.py` first, then reload this page.")
    st.stop()


@st.cache_resource
def resources():
    G = load_graph()
    retriever = Retriever(graph=G)
    retriever.dense_scores("warm up")   # load the embedding model once, not on the first query
    return G, retriever, analyze(G)


@st.cache_resource
def rdf_graph():
    from rdflib import Graph
    g = Graph()
    g.parse(GRAPH_TTL)
    return g


with st.spinner("Getting ready... the first start can take a minute or two."):
    G, retriever, gap_rows = resources()
gap_df = pd.DataFrame(gap_rows)
regions = sorted(node_name(r) for r in nodes_of(G, "Region"))


def passage_text(pid):
    return G.nodes[node_id("Passage", pid)]["text"]


def short_list(items, n=5):
    return ", ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def show_text(text):
    """Render passage text without Markdown turning a leading '1.' into a list."""
    st.markdown(re.sub(r"^(\d+)\.", r"\1\\.", text))


def status_badge(status):
    return f"<span style='color:{STATUS_COLOR[status]};font-weight:600'>● {status}</span>"


def gap_sentence(r):
    """One plain-English line for a region × hazard row."""
    line = f"{status_badge(r.status)} &nbsp; **{r.hazard} in {r.region}** {STATUS_MEANING[r.status]}."
    if r.direct_measures:
        line += f" {r.direct_measures} plan{'s' if r.direct_measures != 1 else ''} found."
    if r.missing_sectors and r.status != "Covered":
        line += f" Missing: {', '.join(r.missing_sectors)}."
    return line


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌍 ClimaGap")
    st.markdown("Reads the climate plans of five Indian states and shows **which weather problems "
                "each plan forgot to fix**. Part of **SDG 13: Climate Action**.")
    st.markdown("**How to use it**\n1. **Ask** a question about the plans.\n"
                "2. **Gaps** shows where plans are missing.\n3. **Map** draws one state's problems and plans.")
    st.caption("A computer read these plans, so it can be wrong. Check the plan pages before you quote a gap.")
    docs = [a for _, a in G.nodes(data=True) if a["type"] == "Doc"]
    with st.expander("Plans used"):
        st.markdown("\n".join(f"- {d['label']}" for d in docs))
        if any("DRAFT" in d["label"] for d in docs):
            st.caption("⚠️ Gujarat's plan is a draft marked \"not to be cited\". Use it for learning only.")
        if any(d.get("kind") == "sample" for d in docs):
            st.warning("These are sample texts written for the demo, not real plan text.")

tab_ask, tab_gaps, tab_map, tab_adv = st.tabs(["🔎 Ask", "🗺️ Gaps", "🕸️ Map", "⚙️ Advanced"])

# ------------------------------------------------------------------ Ask
with tab_ask:
    st.markdown("#### Ask a question about the climate plans")
    st.caption("Tip: name a state and a weather problem, e.g. \"floods in Kerala\".")

    def use_example():
        if st.session_state.example:
            st.session_state.query = st.session_state.example

    st.pills("Try one:", EXAMPLES, key="example", on_change=use_example)
    query = st.text_input("Your question", key="query", placeholder="e.g. heatwave plans in Gujarat")

    with st.expander("Search settings"):
        method = st.radio("How to search", METHODS, index=METHODS.index("hybrid_kg"), format_func=METHOD_LABEL.get)
        k = st.slider("Number of answers", 3, 15, 5)

    if query:
        out = retriever.search(query, method, k)
        if out["dense_error"] and method != "bm25":
            st.info("The meaning-based search isn't available right now, so this uses keywords only.")

        ents = out["query_entities"]
        q_regions = [node_name(e) for e in ents if node_kind(e) == "Region"]
        q_hazards = [node_name(e) for e in ents if node_kind(e) == "Hazard"]
        if q_regions or q_hazards:
            sel = gap_df
            if q_regions:
                sel = sel[sel.region.isin(q_regions)]
            if q_hazards:
                sel = sel[sel.hazard.isin(q_hazards)]
            with st.container(border=True):
                st.markdown("##### Quick answer")
                if sel.empty:
                    st.write("This state isn't listed as facing that weather problem.")
                for r in sel.itertuples():
                    st.markdown(gap_sentence(r), unsafe_allow_html=True)

        st.markdown("##### Pages from the plans")
        for r in out["results"]:
            p = r["passage"]
            with st.container(border=True):
                st.markdown(f"**{p['title']}** · page {p['page']}")
                show_text(p["text"])

# ------------------------------------------------------------------ Gaps
with tab_gaps:
    st.markdown("#### Where are the plans missing?")
    st.caption("Each box is one state and one weather problem. Green = good plan, yellow = half a plan, "
               "red = no clear plan, blank = the state doesn't face that problem.")

    mat = coverage_matrix(gap_rows)
    status_mat = gap_df.pivot(index="region", columns="hazard", values="status").reindex_like(mat).fillna("")
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index), zmin=0, zmax=1,
        colorscale=[[0, "#c62828"], [0.5, "#f9a825"], [1, "#2e7d32"]], showscale=False,
        text=status_mat.values, texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>%{text}<extra></extra>"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)

    gaps = gap_df[gap_df.status != "Covered"].sort_values("priority", ascending=False)
    st.markdown("##### The gaps, biggest first")
    for r in gaps.itertuples():
        st.markdown(gap_sentence(r), unsafe_allow_html=True)

    st.markdown("##### Read the plan pages for a gap")
    pick = st.selectbox("Pick a gap", [f"{r.hazard} in {r.region}" for r in gaps.itertuples()])
    if pick:
        r = gaps[(gaps.hazard + " in " + gaps.region) == pick].iloc[0]
        if r.measures and not r.direct_measures:
            st.markdown("No plan names this problem directly. These pages are the closest the plan gets, "
                        "so read them to decide for yourself:")
            pids = r.evidence
        elif r.measures:
            st.markdown(f"Plans found: {short_list(r.measures)}. "
                        f"**Not covered: {', '.join(r.missing_sectors) or 'nothing'}.**")
            pids = r.evidence
        elif r.mentions:
            st.markdown("The plan talks about this problem, but has no plan for it:")
            pids = r.mentions
        else:
            st.markdown("The plan never mentions this problem, even though the state faces it.")
            pids = []
        for pid in pids[:5]:
            with st.container(border=True):
                show_text(passage_text(pid))

# ------------------------------------------------------------------ Map
with tab_map:
    st.markdown("#### The map of one state")
    region = st.radio("State", regions, horizontal=True)
    st.caption("The state is in the middle. Around it are the weather problems it faces. Around each problem "
               "are the parts of life it hurts (green box = has a plan, red box = no plan) and the top plans "
               "found (green dots). Hover over a dot to read where it comes from.")

    def region_html(region):
        """A fixed radial layout: region → hazards → that hazard's sectors and top measures."""
        from pyvis.network import Network
        net = Network(height="620px", width="100%", cdn_resources="remote")
        rid = node_id("Region", region)
        rows = {r["hazard"]: r for r in gap_rows if r["region"] == region}
        hazards = sorted(out_by_rel(G, rid, "EXPOSED_TO"), key=node_name)
        own_measures = in_by_rel(G, rid, "IN_REGION")
        net.add_node(rid, label=region, color=KIND_COLOR["Region"], shape="star", size=40, x=0, y=0,
                     font={"size": 30, "color": "#1565c0"})
        for i, h in enumerate(hazards):
            name, row = node_name(h), rows[node_name(h)]
            angle = -pi / 2 + 2 * pi * i / len(hazards)
            net.add_node(h, label=f"{name}\n{row['status']}", color=STATUS_COLOR[row["status"]], shape="diamond",
                         size=28, x=230 * cos(angle), y=230 * sin(angle), title=f"{name}: {row['status']}",
                         font={"size": 22})
            net.add_edge(rid, h, color="#90a4ae", width=2)

            covered = set(row["covered_sectors"])
            sectors = sorted(node_name(s) for s in out_by_rel(G, h, "AFFECTS"))
            measures = sorted((m for m in own_measures if addresses(G, m, h)),
                              key=lambda m: -len(G[m][h]["evidence"]))[:MEASURES_PER_HAZARD]
            leaves = [("sector", s) for s in sectors] + [("measure", m) for m in measures]
            span = 2 * pi / len(hazards) * 0.8
            for j, (kind, item) in enumerate(leaves):
                a = angle + (span * (j / (len(leaves) - 1) - 0.5) if len(leaves) > 1 else 0)
                radius = 430 if j % 2 == 0 else 540   # alternate rings so neighbouring labels don't collide
                x, y = radius * cos(a), radius * sin(a)
                if kind == "sector":
                    ok = item in covered
                    leaf = f"{h}|{item}"
                    net.add_node(leaf, label=item, shape="box", x=x, y=y,
                                 color="#2e7d32" if ok else "#c62828", font={"color": "#ffffff", "size": 18},
                                 title=f"{item}: {'has a plan' if ok else 'no plan found'} for {name.lower()}")
                else:
                    label = G.nodes[item]["label"]
                    leaf = f"{h}|{item}"
                    net.add_node(leaf, label=label if len(label) <= 28 else label[:26] + "…", shape="dot",
                                 size=10, x=x, y=y, color=KIND_COLOR["Measure"],
                                 title=passage_text(G[item][h]["evidence"][0])[:300])
                net.add_edge(h, leaf, color="#cfd8dc")
        net.set_options("""{
          "physics": {"enabled": false},
          "interaction": {"dragNodes": false, "hover": true, "zoomView": true, "dragView": true},
          "edges": {"smooth": false},
          "nodes": {"font": {"size": 18}}
        }""")
        # The tab may be hidden when the graph is drawn, so fit it once it has a real size.
        refit = ("<script>const refit = () => { if (window.network) { network.redraw(); network.fit(); } };"
                 "window.addEventListener('resize', refit);"
                 "new ResizeObserver(refit).observe(document.body);</script></body>")
        return net.generate_html().replace("</body>", refit)

    components.html(region_html(region), height=640)

# ------------------------------------------------------------------ Advanced
with tab_adv:
    st.markdown("#### For teachers and examiners")
    kinds = pd.Series([a["type"] for _, a in G.nodes(data=True)]).value_counts()
    st.markdown(f"**Knowledge graph:** {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges  \n"
                + " · ".join(f"{k} {v:,}" for k, v in kinds.items()))

    results_csv = EVAL / "results.csv"
    if results_csv.exists():
        st.markdown("##### How well each search method works")
        st.dataframe(pd.read_csv(results_csv).replace({"method": METHOD_LABEL}), hide_index=True)
        st.caption("P@5, R@10, MRR and nDCG@10 averaged over the labelled queries in eval/queries.json.")

    st.markdown("##### Gap scores")
    st.dataframe(gap_df[["region", "hazard", "severity", "status", "coverage", "priority",
                         "direct_measures", "indirect_measures"]], hide_index=True, width="stretch")
    st.caption("coverage = 0.5 × min(1, direct measures / 5) + 0.5 × share of affected sectors covered; "
               "priority = severity weight × (1 − coverage).")

    st.markdown("##### SPARQL")
    st.caption("The graph is also exported as RDF (data/processed/graph.ttl). Pick an example or write your own query.")
    title = st.selectbox("Example query", list(SPARQL_EXAMPLES))
    q = st.text_area("SPARQL", SPARQL_EXAMPLES[title], height=200, key=f"sparql-{title}")
    if st.button("Run query"):
        try:
            cols, rows = run_sparql(rdf_graph(), q)
            st.dataframe(pd.DataFrame(rows, columns=cols), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    if st.button("Reload data"):
        st.cache_resource.clear()
        st.rerun()
