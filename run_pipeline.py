"""Run the whole ClimaGap pipeline: ingest -> extract -> graph -> gaps (-> evaluate with --eval)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import extract
import gaps
import graph
import ingest

if __name__ == "__main__":
    for stage in (ingest, extract, graph, gaps):
        print(f"\n=== {stage.__name__} ===")
        stage.main()
    if "--eval" in sys.argv:
        import evaluate
        print("\n=== evaluate ===")
        evaluate.main()
