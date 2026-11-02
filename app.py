"""ClimaGap: ask questions over state climate plans, explore the knowledge graph, and see policy gaps.

    streamlit run app.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from common import EVAL, GRAPH_JSON, GRAPH_TTL, node_id, node_kind, node_name
from gaps import STATUS_ORDER, analyze, coverage_matrix
from graph import SPARQL_EXAMPLES, in_by_rel, load_graph, nodes_of, out_by_rel, run_sparql
from retrieve import METHODS, Retriever

st.set_page_config(page_title="ClimaGap", page_icon="🌍", layout="wide")

STATUS_COLOR = {"Covered": "#2e7d32", "Partial": "#f9a825",
                "Weak evidence": "#e65100", "Mentioned, no action": "#ef6c00", "Not mentioned": "#c62828"}
KIND_COLOR = {"Region": "#1565c0", "Hazard": "#c62828", "Sector": "#6d4c41",
              "Measure": "#2e7d32", "SDG": "#00838f"}
METHOD_LABEL = {"bm25": "BM25", "dense": "Dense (MiniLM)", "hybrid": "Hybrid",
                "hybrid_kg": "Hybrid + Knowledge Graph"}

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


with st.spinner("Loading the knowledge graph and embedding model (first start can take a minute)..."):
    G, retriever, gap_rows = resources()
gap_df = pd.DataFrame(gap_rows)


def chip(nid):
    color = KIND_COLOR.get(node_kind(nid), "#555")
    label = G.nodes[nid]["label"] if nid in G else node_name(nid)
    return (f"<span style='background:{color}1a;color:{color};border:1px solid {color}55;"
            f"border-radius:10px;padding:1px 8px;margin:2px;font-size:0.82em;display:inline-block'>{label}</span>")


def passage_text(pid):
    return G.nodes[node_id("Passage", pid)]["text"]


def short_list(items, n=5):
    return ", ".join(items[:n]) + (f" (+{len(items) - n} more)" if len(items) > n else "")


def show_text(text):
    """Render passage text without Markdown turning a leading '1.' into a list."""
    st.markdown(re.sub(r"^(\d+)\.", r"\1\\.", text))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌍 ClimaGap")
    st.caption("A knowledge graph + information retrieval system that finds gaps in state climate "
               "adaptation plans. **SDG 13 · Climate Action**")
    st.markdown("- **13.1** resilience to climate hazards\n- **13.2** climate measures in policy & planning\n"
                "- **13.3** awareness & capacity")
    kinds = pd.Series([a["type"] for _, a in G.nodes(data=True)]).value_counts()
    st.markdown(f"**Graph:** {G.number_of_nodes()} nodes · {G.number_of_edges()} edges  \n"
                + " · ".join(f"{k} {v}" for k, v in kinds.items()))
    if any(a.get("kind") == "sample" for _, a in G.nodes(data=True) if a["type"] == "Doc"):
        st.warning("Running on **sample texts** written for the demo, not official plan text. "
                   "Put real SAPCC PDFs in `data/raw/` and re-run the pipeline for real findings.")
    docs = [a for _, a in G.nodes(data=True) if a["type"] == "Doc"]
    st.markdown("**Sources:**  \n" + "  \n".join(f"- {d['label']}" for d in docs))
    if any("DRAFT" in d["label"] for d in docs):
        st.caption("⚠️ A source is marked as a draft not to be cited. Use its findings for analysis only.")
    st.caption("Gaps are automatic text extraction, so verify any gap against the plan before quoting it. "
               "Hazard severities are indicative.")
    if st.button("Reload data"):
        st.cache_resource.clear()
        st.rerun()

tab_ask, tab_graph, tab_gaps, tab_sparql = st.tabs(["🔎 Ask", "🕸️ Graph", "🗺️ Gap Map", "🧾 SPARQL"])

# ------------------------------------------------------------------ Ask
with tab_ask:
    query = st.text_input("Ask about the climate plans", "heatwave measures in Gujarat")
    c1, c2 = st.columns([3, 1])
    method = c1.radio("Retrieval method", METHODS, index=3, horizontal=True, format_func=METHOD_LABEL.get)
    k = c2.slider("Results", 3, 15, 5)
    if query:
        out = retriever.search(query, method, k)
        if out["dense_error"] and method != "bm25":
            st.info(f"Dense model unavailable, so this uses BM25 only. ({out['dense_error'][:160]})")
        ents = out["query_entities"]
        if method == "hybrid_kg":
            st.markdown("**Entities linked in the graph:** " + (" ".join(chip(e) for e in ents) or "_none_"),
                        unsafe_allow_html=True)

        regions = [node_name(e) for e in ents if node_kind(e) == "Region"]
        hazards = [node_name(e) for e in ents if node_kind(e) == "Hazard"]
        if method == "hybrid_kg" and (regions or hazards):
            with st.container(border=True):
                st.markdown("##### What the knowledge graph says")
                sel = gap_df
                if regions:
                    sel = sel[sel.region.isin(regions)]
                if hazards:
                    sel = sel[sel.hazard.isin(hazards)]
                if sel.empty:
                    st.write("No exposure recorded for this region/hazard combination.")
                for r in sel.itertuples():
                    color = STATUS_COLOR[r.status]
                    measures = short_list(r.measures) if r.measures else "no measure in the plan"
                    missing = f" · sectors not addressed: {', '.join(r.missing_sectors)}" if r.missing_sectors and r.measures else ""
                    st.markdown(f"<span style='color:{color};font-weight:600'>● {r.status}</span> "
                                f"**{r.region} × {r.hazard}** ({r.severity}) → {measures}{missing}",
                                unsafe_allow_html=True)

        for r in out["results"]:
            p = r["passage"]
            with st.container(border=True):
                parts = " · ".join(f"{n} {v:.2f}" for n, v in r["parts"].items())
                st.markdown(f"**#{r['rank']} · {p['title']}** · page {p['page']} · `{p['id']}`  \n"
                            f"<small>score {r['score']:.3f} ({parts})</small>", unsafe_allow_html=True)
                show_text(p["text"])
                st.markdown(" ".join(chip(e) for e in r["entities"] if node_kind(e) != "Measure"),
                            unsafe_allow_html=True)

    results_csv = EVAL / "results.csv"
    if results_csv.exists():
        with st.expander("Retrieval evaluation (eval/queries.json)"):
            st.dataframe(pd.read_csv(results_csv).replace({"method": METHOD_LABEL}), hide_index=True)
            st.caption("P@5, R@10, MRR, nDCG@10 averaged over the labelled queries. Re-run `python src/evaluate.py` to refresh.")

# ------------------------------------------------------------------ Graph
with tab_graph:
    region = st.selectbox("Region", sorted(node_name(r) for r in nodes_of(G, "Region")))
    c1, c2 = st.columns([1, 2])
    show_sdg = c1.checkbox("Show SDG 13 targets", value=False)
    max_measures = c2.slider("Measures shown (most evidenced first)", 5, 100, 30)

    def measure_strength(m):
        """How strongly a measure is supported: evidence passages on its ADDRESSES edges."""
        return sum(len(G[m][t]["evidence"]) for t in out_by_rel(G, m, "ADDRESSES"))

    def region_html(region):
        from pyvis.network import Network
        net = Network(height="640px", width="100%", directed=True, cdn_resources="remote")
        rid = node_id("Region", region)
        status = {r["hazard"]: r["status"] for r in gap_rows if r["region"] == region}
        net.add_node(rid, label=region, color=KIND_COLOR["Region"], shape="star", size=35)
        for h in out_by_rel(G, rid, "EXPOSED_TO"):
            name = node_name(h)
            net.add_node(h, label=f"{name}\n{status[name]}", color=STATUS_COLOR[status[name]], shape="diamond",
                         size=26, title=f"{name}: {status[name]}")
            net.add_edge(rid, h, label=G[rid][h]["severity"], title="EXPOSED_TO", color="#90a4ae")
            for s in out_by_rel(G, h, "AFFECTS"):
                if s not in net.get_nodes():
                    net.add_node(s, label=node_name(s), color=KIND_COLOR["Sector"], shape="box",
                                 font={"color": "#ffffff"})
                net.add_edge(h, s, title="AFFECTS", dashes=True, color="#bcaaa4")
        measures = sorted(in_by_rel(G, rid, "IN_REGION"), key=measure_strength, reverse=True)[:max_measures]
        for m in measures:
            a = G.nodes[m]
            targets = out_by_rel(G, m, "ADDRESSES")
            evidence = G[m][targets[0]]["evidence"] if targets else []
            tooltip = passage_text(evidence[0])[:300] if evidence else a["label"]
            net.add_node(m, label=a["label"], color=KIND_COLOR["Measure"], shape="dot", size=12, title=tooltip)
            linked = False
            for t in targets:
                if t in net.get_nodes():
                    net.add_edge(m, t, title="ADDRESSES", color="#81c784")
                    linked = True
            if not linked:
                net.add_edge(m, rid, title="IN_REGION", color="#cfd8dc")
            if show_sdg:
                for sdg in out_by_rel(G, m, "CONTRIBUTES_TO"):
                    if sdg not in net.get_nodes():
                        net.add_node(sdg, label=G.nodes[sdg]["label"], color=KIND_COLOR["SDG"], shape="hexagon")
                    net.add_edge(m, sdg, title="CONTRIBUTES_TO", color="#80deea")
        net.set_options("""{
          "physics": {"barnesHut": {"gravitationalConstant": -2500, "springLength": 95, "avoidOverlap": 0.3},
                      "stabilization": {"iterations": 400, "fit": true}},
          "edges": {"smooth": false, "font": {"size": 11}},
          "nodes": {"font": {"size": 14}}
        }""")
        # The tab may be hidden while vis lays out the graph, so re-fit once it has a real size.
        refit = ("<script>const refit = () => { if (window.network) { network.redraw(); network.fit(); } };"
                 "window.addEventListener('resize', refit);"
                 "new ResizeObserver(refit).observe(document.body);</script></body>")
        return net.generate_html().replace("</body>", refit)

    components.html(region_html(region), height=660)
    st.markdown(" ".join(f"<span style='color:{c}'>◆ {s}</span>" for s, c in STATUS_COLOR.items())
                + f" &nbsp; <span style='color:{KIND_COLOR['Measure']}'>● policy measure</span>"
                  f" &nbsp; <span style='color:{KIND_COLOR['Sector']}'>■ sector</span>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------ Gap Map
with tab_gaps:
    counts = gap_df.status.value_counts()
    cols = st.columns(len(STATUS_ORDER))
    for col, s in zip(cols, STATUS_ORDER):
        col.metric(s, int(counts.get(s, 0)))

    mat = coverage_matrix(gap_rows)
    status_mat = gap_df.pivot(index="region", columns="hazard", values="status").reindex_like(mat).fillna("")
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=list(mat.columns), y=list(mat.index), zmin=0, zmax=1,
        colorscale=[[0, "#c62828"], [0.5, "#f9a825"], [1, "#2e7d32"]],
        text=status_mat.values, texttemplate="%{text}", colorbar=dict(title="coverage"),
        hovertemplate="%{y} × %{x}<br>coverage %{z}<br>%{text}<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title="Policy coverage: region × hazard (blank = not exposed)")
    st.plotly_chart(fig)

    gaps = gap_df[gap_df.status != "Covered"].sort_values("priority", ascending=False)
    view = gaps.assign(missing_sectors=gaps.missing_sectors.map(", ".join),
                       measures=gaps.measures.map(short_list))
    st.markdown("##### Gaps ranked by priority (severity × uncovered share)")
    st.dataframe(view[["region", "hazard", "severity", "status", "coverage", "priority", "missing_sectors", "measures"]],
                 hide_index=True, width="stretch")

    pick = st.selectbox("Evidence for", [f"{r.region} × {r.hazard}" for r in gaps.itertuples()])
    if pick:
        r = gaps[(gaps.region + " × " + gaps.hazard) == pick].iloc[0]
        st.markdown(f"**Status:** <span style='color:{STATUS_COLOR[r.status]}'>{r.status}</span>",
                    unsafe_allow_html=True)
        if r.measures and not r.direct_measures:
            st.markdown(f"No measure names this hazard in its own clause. Only **{r.indirect_measures} indirect** "
                        f"link(s), inferred from the surrounding passage: {short_list(r.measures)}. "
                        "Check the passages below to see whether the plan frames any action around this hazard.")
            for pid in r.evidence[:5]:
                st.caption(f"`{pid}`")
                show_text(passage_text(pid))
        elif r.measures:
            st.markdown(f"**{r.direct_measures} direct** measures ({r.indirect_measures} indirect), e.g. "
                        f"{short_list(r.measures)}. Sectors covered: {', '.join(r.covered_sectors) or '-'}. "
                        f"**Not addressed: {', '.join(r.missing_sectors) or '-'}**")
            for pid in r.evidence[:5]:
                st.caption(f"`{pid}`")
                show_text(passage_text(pid))
        elif r.mentions:
            st.markdown("The plan **mentions** this hazard but proposes no measure for it:")
            for pid in r.mentions[:5]:
                st.caption(f"`{pid}`")
                show_text(passage_text(pid))
        else:
            st.markdown("The plan never mentions this hazard, although the region is exposed to it.")

# ------------------------------------------------------------------ SPARQL
with tab_sparql:
    st.caption("The graph is also exported as RDF (`data/processed/graph.ttl`). Pick an example or write your own query.")
    title = st.selectbox("Example query", list(SPARQL_EXAMPLES))
    q = st.text_area("SPARQL", SPARQL_EXAMPLES[title], height=220, key=f"sparql-{title}")
    if st.button("Run query"):
        try:
            cols, rows = run_sparql(rdf_graph(), q)
            st.dataframe(pd.DataFrame(rows, columns=cols), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
