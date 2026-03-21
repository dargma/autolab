"""Markdown file parsers and updaters for TRACKER, REGISTRY, DECISIONS."""

import re
from pathlib import Path
from datetime import date


class TrackerMD:
    """Parse and update TRACKER.md experiment status matrix."""

    def __init__(self, path):
        self.path = Path(path)

    def read_rows(self):
        """Return list of dicts with keys: num, name, status, metric, date, notes."""
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().split("\n"):
            line = line.strip()
            if not line.startswith("|") or line.startswith("| #") or line.startswith("|--"):
                continue
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 6 and parts[0].isdigit():
                rows.append({
                    "num": int(parts[0]),
                    "name": parts[1],
                    "status": parts[2],
                    "metric": parts[3],
                    "date": parts[4],
                    "notes": parts[5],
                })
        return rows

    def next_number(self):
        """Return the next experiment number."""
        rows = self.read_rows()
        return max((r["num"] for r in rows), default=0) + 1

    def get_best_result(self):
        """Return the best result row (highest accuracy from metric field)."""
        rows = self.read_rows()
        best = None
        best_acc = -1
        for r in rows:
            if "done" not in r["status"]:
                continue
            match = re.search(r"([\d.]+)%", r["metric"])
            if match:
                acc = float(match.group(1))
                if acc > best_acc:
                    best_acc = acc
                    best = r
        return best

    def add_row(self, num, name, status, metric, dt, notes):
        """Append a new row to TRACKER.md."""
        text = self.path.read_text() if self.path.exists() else ""
        row = f"| {num:03d} | {name} | {status} | {metric} | {dt} | {notes} |"
        self.path.write_text(text.rstrip() + "\n" + row + "\n")

    def update_row(self, num, status=None, metric=None, notes=None):
        """Update an existing row by experiment number."""
        if not self.path.exists():
            return
        lines = self.path.read_text().split("\n")
        for i, line in enumerate(lines):
            match = re.match(r"\|\s*0*(\d+)\s*\|", line)
            if match and int(match.group(1)) == num:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if status is not None:
                    parts[2] = status
                if metric is not None:
                    parts[3] = metric
                if notes is not None:
                    parts[5] = notes
                lines[i] = "| " + " | ".join(parts) + " |"
                break
        self.path.write_text("\n".join(lines))


class RegistryMD:
    """Parse and append to REGISTRY.md sections."""

    def __init__(self, path):
        self.path = Path(path)

    def append_to_section(self, section, entry):
        """Append an entry line to a section (e.g., '## Established Facts')."""
        if not self.path.exists():
            return
        text = self.path.read_text()
        marker = f"## {section}"
        if marker not in text:
            text += f"\n\n{marker}\n- {entry}\n"
        else:
            # Find the section and append before the next section or EOF
            lines = text.split("\n")
            insert_idx = len(lines)
            in_section = False
            for i, line in enumerate(lines):
                if line.strip().startswith(marker):
                    in_section = True
                    continue
                if in_section and line.strip().startswith("## "):
                    insert_idx = i
                    break
            # Insert before next section
            lines.insert(insert_idx, f"- {entry}")
            text = "\n".join(lines)
        self.path.write_text(text)


class DecisionsMD:
    """Append direction change entries to DECISIONS.md."""

    def __init__(self, path):
        self.path = Path(path)

    def add_decision(self, title, reason, before, after, impact):
        """Append a new decision entry."""
        text = self.path.read_text() if self.path.exists() else ""
        today = date.today().isoformat()
        entry = f"""
### {today}: {title}
- Reason: {reason}
- Before: {before}
- After: {after}
- Impact: {impact}
"""
        self.path.write_text(text.rstrip() + "\n" + entry)
