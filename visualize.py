"""
Generate visualizations from optimization results.
Run after optimizer.py completes.

Usage:
    python visualize.py
"""

import json
from pathlib import Path


def generate_report():
    """Generate a markdown report from optimization results."""
    results_dir = Path("results")
    report_path = results_dir / "final_report.json"

    if not report_path.exists():
        print("No results found. Run optimizer.py first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    summary = report["optimization_summary"]
    curve = report["improvement_curve"]
    clusters = report.get("failure_clusters", {})

    # ASCII improvement curve
    print("\n" + "=" * 60)
    print("OPTIMIZATION RESULTS")
    print("=" * 60)

    print(f"\n📊 Summary:")
    print(f"  Total calls:   {summary['total_calls']}")
    print(f"  Total cost:    ${summary['total_cost']:.2f}")
    print(f"  Start score:   {summary['starting_score']:.3f}")
    print(f"  Best score:    {summary['best_score']:.3f}")
    print(f"  Improvement:   +{summary['improvement']:.3f} ({summary['improvement_pct']:.0f}%)")

    # Improvement curve
    print(f"\n📈 Improvement Curve:")
    print(f"  {'Phase':<7} {'Iter':<6} {'Composite':<12} {'Bar'}")
    print(f"  {'─'*55}")
    for point in curve:
        bar_len = int(point["avg_composite"] * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  P{point['phase']:<6} {point['iteration']:<6} {point['avg_composite']:<12.3f} {bar}")

    # Failure clusters
    if clusters.get("clusters"):
        print(f"\n🔍 Failure Clusters:")
        for c in clusters["clusters"]:
            cluster_id = c.get("id", c.get("cluster_id", "?"))
            avg_score = c.get("avg_checklist", c.get("avg_checklist_score", "N/A"))
            print(f"  Cluster {cluster_id}: {c['size']} calls, avg checklist={avg_score}")
            if c.get("top_terms"):
                print(f"    Top terms: {', '.join(c['top_terms'])}")
            if c.get("avg_hedges") is not None:
                print(f"    Avg hedges: {c['avg_hedges']}")

    # Before/after prompts
    print(f"\n📝 Starting Prompt:")
    print(f"  {summary['starting_prompt']}")

    print(f"\n📝 Best Prompt:")
    for line in summary["best_prompt"].split("\n"):
        print(f"  {line}")

    print(f"\n📝 Best First Message:")
    print(f"  {summary['best_first_message']}")

    # Per-iteration details
    if "full_history" in report:
        print(f"\n📋 Per-Iteration Scores:")
        for h in report["full_history"]:
            scores_detail = []
            for s in h.get("scores", []):
                sd = s.get("structured_data", {})
                checks = sum([
                    sd.get("schedulerGreetedProperly", False),
                    sd.get("schedulerCollectedName", False),
                    sd.get("schedulerOfferedTimes", False),
                    sd.get("schedulerProvidedPricing", False),
                    sd.get("schedulerConfirmedAppointment", False),
                    sd.get("appointmentBooked", False),
                ])
                scores_detail.append(f"{checks}/6")

            avg_cl = h.get('avg_checklist')
            cl_str = f"{avg_cl:.3f}" if isinstance(avg_cl, (int, float)) else "N/A"
            print(f"  P{h['phase']} Iter {h['iteration']}: composite={h['avg_composite']:.3f} checklist={cl_str} calls=[{', '.join(scores_detail)}]")


if __name__ == "__main__":
    generate_report()
