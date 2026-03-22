"""Tests for autolab.knowledge — markdown parsers for TRACKER, REGISTRY."""

import pytest

from autolab.knowledge import TrackerMD, RegistryMD


class TestTrackerMD:
    """Test TrackerMD parsing and row operations."""

    SAMPLE_TRACKER = (
        "# Experiment Tracker\n\n"
        "| # | Name | Status | Key Metric | Date | Notes |\n"
        "|---|------|--------|------------|------|-------|\n"
        "| 001 | baseline | done | 95.00% | 2026-01-01 | initial run |\n"
        "| 002 | cnn-v1 | done | 97.50% | 2026-01-05 | improved |\n"
    )

    def test_parse_rows(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text(self.SAMPLE_TRACKER)
        tracker = TrackerMD(tracker_file)
        rows = tracker.read_rows()
        assert len(rows) == 2
        assert rows[0]["num"] == 1
        assert rows[0]["name"] == "baseline"
        assert rows[0]["metric"] == "95.00%"
        assert rows[1]["num"] == 2
        assert rows[1]["name"] == "cnn-v1"

    def test_next_number(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text(self.SAMPLE_TRACKER)
        tracker = TrackerMD(tracker_file)
        assert tracker.next_number() == 3

    def test_next_number_empty(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text("# Empty Tracker\n")
        tracker = TrackerMD(tracker_file)
        assert tracker.next_number() == 1

    def test_add_row(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text(self.SAMPLE_TRACKER)
        tracker = TrackerMD(tracker_file)
        tracker.add_row(3, "new-exp", "planned", "-", "2026-03-22", "testing")
        rows = tracker.read_rows()
        assert len(rows) == 3
        assert rows[2]["num"] == 3
        assert rows[2]["name"] == "new-exp"
        assert rows[2]["status"] == "planned"

    def test_get_best_result(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text(self.SAMPLE_TRACKER)
        tracker = TrackerMD(tracker_file)
        best = tracker.get_best_result()
        assert best is not None
        assert best["num"] == 2
        assert "97.50%" in best["metric"]

    def test_update_row(self, tmp_path):
        tracker_file = tmp_path / "TRACKER.md"
        tracker_file.write_text(self.SAMPLE_TRACKER)
        tracker = TrackerMD(tracker_file)
        tracker.update_row(1, status="failed", metric="0%")
        rows = tracker.read_rows()
        assert rows[0]["status"] == "failed"
        assert rows[0]["metric"] == "0%"

    def test_nonexistent_file_returns_empty(self, tmp_path):
        tracker = TrackerMD(tmp_path / "missing.md")
        assert tracker.read_rows() == []


class TestRegistryMD:
    """Test RegistryMD section parsing and appending."""

    SAMPLE_REGISTRY = (
        "# Registry\n\n"
        "## Established Facts\n"
        "- Baseline accuracy is 95%\n\n"
        "## Hypotheses\n"
        "- CNN should beat FC\n\n"
        "## Rejected Ideas\n\n"
        "## Open Questions\n"
        "- What about data augmentation?\n"
    )

    def test_append_to_existing_section(self, tmp_path):
        reg_file = tmp_path / "REGISTRY.md"
        reg_file.write_text(self.SAMPLE_REGISTRY)
        reg = RegistryMD(reg_file)
        reg.append_to_section("Established Facts", "BatchNorm helps by 0.5%")
        content = reg_file.read_text()
        assert "BatchNorm helps by 0.5%" in content
        # Original entry should still be there
        assert "Baseline accuracy is 95%" in content

    def test_append_to_new_section(self, tmp_path):
        reg_file = tmp_path / "REGISTRY.md"
        reg_file.write_text("# Registry\n")
        reg = RegistryMD(reg_file)
        reg.append_to_section("New Section", "First entry")
        content = reg_file.read_text()
        assert "## New Section" in content
        assert "- First entry" in content

    def test_append_preserves_other_sections(self, tmp_path):
        reg_file = tmp_path / "REGISTRY.md"
        reg_file.write_text(self.SAMPLE_REGISTRY)
        reg = RegistryMD(reg_file)
        reg.append_to_section("Hypotheses", "Residual connections may help")
        content = reg_file.read_text()
        # Both original and new entries present
        assert "CNN should beat FC" in content
        assert "Residual connections may help" in content
        # Other sections untouched
        assert "## Established Facts" in content
        assert "## Open Questions" in content
