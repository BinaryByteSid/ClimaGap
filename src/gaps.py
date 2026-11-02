"""Stage 5: gap analysis — compare each region's hazard exposure with the measures in its own plan.

For every Region -EXPOSED_TO-> Hazard:
  * direct measures = region's measures whose own clause/sentence names the hazard
                      (links inferred only from the surrounding passage are "indirect")
  * sectors         = sectors the hazard AFFECTS; covered if a direct measure also directly addresses it
  * coverage        = 0.5 x min(1, direct / STRONG) + 0.5 x share of affected sectors covered
  * status          = Covered | Partial | Weak evidence | Mentioned, no action | Not mentioned
  * priority        = severity weight x (1 - coverage)
"""
import csv

from common import GAPS_CSV, node_name
from graph import in_by_rel, load_graph, nodes_of, out_by_rel

SEVERITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}
STATUS_ORDER = ["Not mentioned", "Mentioned, no action", "Weak evidence", "Partial", "Covered"]
STRONG = 5   # direct measures needed before a hazard counts as fully addressed


def addresses(G, m, target, direct_only=True):
    if not G.has_edge(m, target) or G[m][target]["rel"] != "ADDRESSES":
        return False
    return not direct_only or G[m][target].get("level") != "passage"


def plan_passages(G, region):
    docs = out_by_rel(G, region, "HAS_PLAN")
    return {p for d in docs for p in in_by_rel(G, d, "PART_OF")}


def analyze(G):
    rows = []
    for region in sorted(nodes_of(G, "Region")):
        own_passages = plan_passages(G, region)
        own_measures = in_by_rel(G, region, "IN_REGION")
        for hazard in sorted(out_by_rel(G, region, "EXPOSED_TO")):
            severity = G[region][hazard]["severity"]
            affected = set(out_by_rel(G, hazard, "AFFECTS"))
            measures = [m for m in own_measures if addresses(G, m, hazard)]
            indirect = [m for m in own_measures if addresses(G, m, hazard, direct_only=False) and m not in measures]
            covered = {s for m in measures for s in affected if addresses(G, m, s)}
            mentions = sorted(p for p in out_by_rel(G, hazard, "MENTIONED_IN") if p in own_passages)
            if measures:
                share = len(covered) / len(affected) if affected else 1
                coverage = 0.5 * min(1, len(measures) / STRONG) + 0.5 * share
                status = "Covered" if covered == affected and len(measures) >= STRONG else "Partial"
            else:
                coverage = 0.0
                status = ("Weak evidence" if indirect else
                          "Mentioned, no action" if mentions else "Not mentioned")
            measures.sort(key=lambda m: -len(G[m][hazard]["evidence"]))
            evidence = sorted({e for m in measures or indirect for e in G[m][hazard]["evidence"]})
            rows.append({
                "region": node_name(region),
                "hazard": node_name(hazard),
                "severity": severity,
                "status": status,
                "coverage": round(coverage, 2),
                "priority": round(SEVERITY_WEIGHT.get(severity, 1) * (1 - coverage), 2),
                "direct_measures": len(measures),
                "indirect_measures": len(indirect),
                "measures": [G.nodes[m]["label"] for m in measures or indirect],
                "covered_sectors": sorted(node_name(s) for s in covered),
                "missing_sectors": sorted(node_name(s) for s in affected - covered),
                "evidence": evidence,
                "mentions": [node_name(p) for p in mentions],
            })
    return rows


def coverage_matrix(rows):
    import pandas as pd
    df = pd.DataFrame(rows)
    return df.pivot(index="region", columns="hazard", values="coverage")


def save(rows, path=GAPS_CSV):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: "; ".join(v) if isinstance(v, list) else v for k, v in r.items()})


def main():
    rows = analyze(load_graph())
    save(rows)
    print(f"Coverage matrix (0 = no direct measure, 1 = {STRONG}+ direct measures and all affected sectors):\n")
    print(coverage_matrix(rows).fillna("-").to_string())
    print("\nTop gaps by priority:")
    for r in sorted(rows, key=lambda r: -r["priority"])[:12]:
        if r["status"] == "Covered":
            continue
        missing = f" missing: {', '.join(r['missing_sectors'])}" if r["missing_sectors"] and r["direct_measures"] else ""
        print(f"  {r['priority']:>4}  {r['region']:<12} {r['hazard']:<15} {r['severity']:<7} "
              f"{r['status']:<21} direct={r['direct_measures']} indirect={r['indirect_measures']}{missing}")
    print(f"\nSaved {len(rows)} rows -> {GAPS_CSV.name}")


if __name__ == "__main__":
    main()
