"""Disk safety guard and system checks."""

import subprocess


def check_disk(path=".", threshold=95):
    """Return True if disk usage is below threshold, False otherwise.

    Prints a warning and returns False if usage >= threshold.
    """
    try:
        result = subprocess.run(
            ["df", "-h", path],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.strip().split("\n")[1:]:
            parts = line.split()
            usage_str = [p for p in parts if p.endswith("%")]
            if usage_str:
                usage = int(usage_str[0].rstrip("%"))
                if usage >= threshold:
                    print(f"DISK {usage}% — All operations halted. Cleanup required.")
                    return False
                return True
    except Exception as e:
        print(f"Warning: Could not check disk usage: {e}")
    return True
