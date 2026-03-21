"""Dashboard HTML generator — single-file visualization with Chart.js."""

import json
from pathlib import Path
from datetime import datetime

from .knowledge import TrackerMD, DecisionsMD
from .safety import check_disk


def generate_dashboard(project_dir, output_path=None):
    """Generate dashboard.html for a project directory."""
    project_dir = Path(project_dir)
    if output_path is None:
        output_path = project_dir / "dashboard.html"

    # Load tracker data
    tracker_path = project_dir / "experiments" / "TRACKER.md"
    tracker = TrackerMD(tracker_path)
    rows = tracker.read_rows()

    # Load ralph log if exists
    ralph_log_path = project_dir / "ralph-log.json"
    ralph_iterations = []
    if ralph_log_path.exists():
        with open(ralph_log_path) as f:
            ralph_iterations = json.load(f)

    # Load all experiment results for charts
    exp_names = []
    exp_accs = []
    exp_lats = []
    exp_params = []
    experiments_dir = project_dir / "experiments"
    for row in rows:
        exp_dir = None
        for d in sorted(experiments_dir.iterdir()):
            if d.is_dir() and d.name.startswith(f"exp-{row['num']:03d}"):
                exp_dir = d
                break
        if exp_dir:
            summary_path = exp_dir / "results" / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
                for r in summary.get("ranking", []):
                    exp_names.append(f"[{row['num']:03d}] {r['name']}")
                    exp_accs.append(r["accuracy"] * 100)
                    exp_lats.append(r["avg_latency_ms"])
                    exp_params.append(r["params"])

    # Load goal if exists
    goal_path = project_dir / "goal.yaml"
    target_acc = 95.0
    target_lat = 33.3
    if goal_path.exists():
        import yaml
        with open(goal_path) as f:
            goal = yaml.safe_load(f)
        metrics = goal.get("metrics", {})
        if "accuracy" in metrics:
            target_acc = metrics["accuracy"]["target"] * 100
        if "avg_latency_ms" in metrics:
            target_lat = metrics["avg_latency_ms"]["target"]

    # Decisions
    decisions_path = project_dir / "knowledge" / "DECISIONS.md"
    decisions_text = ""
    if decisions_path.exists():
        decisions_text = decisions_path.read_text()

    # Disk usage
    disk_ok = check_disk(str(project_dir))

    # Build HTML
    tracker_rows_html = ""
    for row in rows:
        status_class = {
            "done": "done", "running": "running",
            "planned": "planned", "failed": "failed"
        }
        cls = ""
        for key, val in status_class.items():
            if key in row["status"]:
                cls = val
                break
        tracker_rows_html += f"""
        <tr class="{cls}">
            <td>{row['num']:03d}</td>
            <td>{row['name']}</td>
            <td>{row['status']}</td>
            <td>{row['metric']}</td>
            <td>{row['date']}</td>
            <td>{row['notes']}</td>
        </tr>"""

    ralph_rows_html = ""
    for it in ralph_iterations:
        ralph_rows_html += f"""
        <tr>
            <td>{it.get('iteration', '?')}</td>
            <td>{it.get('strategy', 'N/A')}</td>
            <td>{it.get('gap', 'N/A')}</td>
            <td>{it.get('result', 'N/A')}</td>
            <td>{it.get('reasoning', '')}</td>
        </tr>"""

    # Best result
    best_name = "N/A"
    best_acc = 0
    best_lat = 0
    if exp_accs:
        best_idx = exp_accs.index(max(exp_accs))
        best_name = exp_names[best_idx]
        best_acc = exp_accs[best_idx]
        best_lat = exp_lats[best_idx]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoLab Dashboard — {project_dir.name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 20px; }}
  h1 {{ text-align: center; margin-bottom: 20px; color: #1a1a2e; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1400px; margin: 0 auto; }}
  .panel {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .panel h2 {{ font-size: 16px; color: #555; margin-bottom: 12px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  .full-width {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f8f8f8; font-weight: 600; }}
  tr.done td:nth-child(3) {{ color: #27ae60; font-weight: bold; }}
  tr.running td:nth-child(3) {{ color: #f39c12; font-weight: bold; }}
  tr.failed td:nth-child(3) {{ color: #e74c3c; font-weight: bold; }}
  .best-card {{ text-align: center; padding: 30px; }}
  .best-card .value {{ font-size: 48px; font-weight: bold; color: #27ae60; }}
  .best-card .label {{ font-size: 14px; color: #777; margin-top: 4px; }}
  .best-card .name {{ font-size: 18px; margin-top: 8px; }}
  .status-bar {{ display: flex; gap: 20px; justify-content: center; align-items: center; }}
  .status-item {{ text-align: center; }}
  .status-item .val {{ font-size: 24px; font-weight: bold; }}
  .disk-ok {{ color: #27ae60; }}
  .disk-warn {{ color: #e74c3c; }}
  canvas {{ max-height: 350px; }}
</style>
</head>
<body>

<h1>AutoLab Dashboard &mdash; {project_dir.name}</h1>

<div class="grid">

  <!-- Best Result -->
  <div class="panel best-card">
    <h2>Current Best Result</h2>
    <div class="value">{best_acc:.2f}%</div>
    <div class="label">Test Accuracy</div>
    <div class="name">{best_name}</div>
    <div class="label">{best_lat:.2f}ms latency</div>
  </div>

  <!-- Resource Status -->
  <div class="panel">
    <h2>Resource Status</h2>
    <div class="status-bar">
      <div class="status-item">
        <div class="val">{len(rows)}</div>
        <div class="label">Experiments</div>
      </div>
      <div class="status-item">
        <div class="val">{sum(1 for r in rows if 'done' in r['status'])}</div>
        <div class="label">Completed</div>
      </div>
      <div class="status-item">
        <div class="val {'disk-ok' if disk_ok else 'disk-warn'}">{'OK' if disk_ok else 'WARNING'}</div>
        <div class="label">Disk Status</div>
      </div>
      <div class="status-item">
        <div class="val">{len(ralph_iterations)}</div>
        <div class="label">Ralph Iterations</div>
      </div>
    </div>
    <br>
    <div style="text-align:center;color:#999;font-size:12px;">
      Target: {target_acc:.0f}% accuracy, {target_lat:.1f}ms latency
    </div>
  </div>

  <!-- Experiment Tracker -->
  <div class="panel full-width">
    <h2>Experiment Tracker</h2>
    <table>
      <tr><th>#</th><th>Name</th><th>Status</th><th>Key Metric</th><th>Date</th><th>Notes</th></tr>
      {tracker_rows_html}
    </table>
  </div>

  <!-- Accuracy Chart -->
  <div class="panel">
    <h2>Model Accuracy Comparison</h2>
    <canvas id="accChart"></canvas>
  </div>

  <!-- Pareto Chart -->
  <div class="panel">
    <h2>Pareto: Accuracy vs Latency</h2>
    <canvas id="paretoChart"></canvas>
  </div>

  <!-- Ralph-Loop Log -->
  <div class="panel full-width">
    <h2>Ralph-Loop Iteration Log</h2>
    {"<table><tr><th>Iter</th><th>Strategy</th><th>Gap</th><th>Result</th><th>Reasoning</th></tr>" + ralph_rows_html + "</table>" if ralph_iterations else "<p style='color:#999;text-align:center;'>Ralph-loop iterations will appear here after running.</p>"}
  </div>

</div>

<script>
const names = {json.dumps(exp_names)};
const accs = {json.dumps(exp_accs)};
const lats = {json.dumps(exp_lats)};
const targetAcc = {target_acc};

// Accuracy bar chart
new Chart(document.getElementById('accChart'), {{
  type: 'bar',
  data: {{
    labels: names,
    datasets: [{{
      label: 'Accuracy (%)',
      data: accs,
      backgroundColor: accs.map(a => a >= targetAcc ? '#27ae60' : '#3498db'),
      borderWidth: 1
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      annotation: {{ annotations: {{ target: {{ type: 'line', yMin: targetAcc, yMax: targetAcc, borderColor: 'red', borderDash: [5,5] }} }} }}
    }},
    scales: {{
      x: {{ ticks: {{ maxRotation: 45 }} }},
      y: {{ beginAtZero: false, min: Math.max(0, Math.min(...accs) - 5) }}
    }}
  }}
}});

// Pareto scatter
new Chart(document.getElementById('paretoChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{
      label: 'Models',
      data: names.map((n, i) => ({{ x: lats[i], y: accs[i] }})),
      backgroundColor: '#3498db',
      pointRadius: 6,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Latency (ms)' }} }},
      y: {{ title: {{ display: true, text: 'Accuracy (%)' }}, beginAtZero: false }}
    }},
    plugins: {{
      tooltip: {{
        callbacks: {{
          label: function(ctx) {{
            return names[ctx.dataIndex] + ': ' + accs[ctx.dataIndex].toFixed(2) + '%, ' + lats[ctx.dataIndex].toFixed(2) + 'ms';
          }}
        }}
      }}
    }}
  }}
}});
</script>

<footer style="text-align:center;color:#999;font-size:11px;margin-top:20px;">
  Generated by autolab v0.1.0 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</footer>

</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Dashboard saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m autolab.dashboard <project_dir>")
        sys.exit(1)
    generate_dashboard(sys.argv[1])
