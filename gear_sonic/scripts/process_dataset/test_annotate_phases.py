"""Storage regression tests; no Qt or real dataset writes required."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from annotate_phases import CAMERA, Dataset, PHASES, format_time, phase_values


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "meta").mkdir()
        self.path = self.root / "meta/interaction_metadata.jsonl"
        self.records = [
            {"episode_index": 2, "person": "Test", "phase_timestamps": dict.fromkeys(PHASES, 0.0)},
            {"episode_index": 17, "custom": {"keep": [1, 2]},
             "phase_timestamps": {**dict.fromkeys(PHASES, 1.25), "extra": 7}},
        ]
        self.original = (json.dumps(self.records[0], ensure_ascii=False) + "\n\n"
                         + json.dumps(self.records[1]) + "\n").encode()
        self.path.write_bytes(self.original)
        for episode, chunk in ((2, 0), (17, 1)):
            folder = self.root / f"videos/chunk-{chunk:03d}" / CAMERA
            folder.mkdir(parents=True)
            (folder / f"episode_{episode:06d}.mp4").touch()

    def test_discovers_chunks_and_non_contiguous_ids_without_writing(self):
        dataset = Dataset(self.root)
        self.assertEqual(dataset.episodes, [2, 17])
        self.assertEqual(set(dataset.videos), {2, 17})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertFalse(self.path.with_suffix(".jsonl.bak").exists())

    def test_save_preserves_other_lines_fields_and_original_backup(self):
        dataset = Dataset(self.root)
        values = dict(zip(PHASES, (0.5, 1.2, None, None)))
        dataset.save(2, values)
        reloaded = Dataset(self.root)
        self.assertEqual(phase_values(reloaded.records[2]), values)
        self.assertEqual(reloaded.records[2]["phase_timestamps"]["release_start"], 0.0)
        self.assertEqual(reloaded.records[2]["phase_timestamps"]["idle_start"], 0.0)
        self.assertNotIn(None, reloaded.records[2]["phase_timestamps"].values())
        self.assertEqual(reloaded.records[2]["person"], "Test")
        self.assertEqual(reloaded.lines[1:], self.original.splitlines(keepends=True)[1:])
        self.assertEqual(self.path.with_suffix(".jsonl.bak").read_bytes(), self.original)
        dataset.save(17, dict.fromkeys(PHASES, 2.0))
        self.assertEqual(Dataset(self.root).records[17]["phase_timestamps"]["extra"], 7)
        self.assertEqual(Dataset(self.root).records[17]["custom"], {"keep": [1, 2]})
        self.assertEqual(self.path.with_suffix(".jsonl.bak").read_bytes(), self.original)
        self.assertEqual(list(self.path.parent.glob(".phases-*")), [])

    def test_reset_and_resume_partial_annotation(self):
        dataset = Dataset(self.root)
        dataset.save(2, dict(zip(PHASES, (0.5, 1.0, None, None))))
        values = phase_values(Dataset(self.root).records[2])
        self.assertEqual(values[PHASES[0]], 0.5)
        self.assertEqual(values[PHASES[1]], 1.0)
        self.assertIsNone(values[PHASES[2]])
        dataset.save(2, dict.fromkeys(PHASES))
        reset_record = Dataset(self.root).records[2]
        self.assertTrue(all(v is None for v in phase_values(reset_record).values()))
        self.assertEqual({reset_record["phase_timestamps"][phase] for phase in PHASES}, {0.0})
        self.assertNotIn("_phase_annotation_progress", reset_record)

    def test_removed_middle_phase_is_saved_as_zero(self):
        dataset = Dataset(self.root)
        values = dict(zip(PHASES, (1.0, None, 3.0, 4.0)))
        dataset.save(2, values)
        saved = Dataset(self.root).records[2]
        self.assertEqual(saved["phase_timestamps"]["contact_active_start"], 0.0)
        self.assertEqual(phase_values(saved), values)

    def test_old_progress_marker_is_removed_on_save(self):
        dataset = Dataset(self.root)
        dataset.records[2]["_phase_annotation_progress"] = 3
        dataset.save(2, dict(zip(PHASES, (1.0, 2.0, 3.0, None))))
        saved = Dataset(self.root).records[2]
        self.assertNotIn("_phase_annotation_progress", saved)
        self.assertEqual(phase_values(saved), dict(zip(PHASES, (1.0, 2.0, 3.0, None))))

    def test_stale_instance_does_not_overwrite_new_data(self):
        first, second = Dataset(self.root), Dataset(self.root)
        first.save(2, dict.fromkeys(PHASES, 1.0))
        saved = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Another process"):
            second.save(17, dict.fromkeys(PHASES, 2.0))
        self.assertEqual(self.path.read_bytes(), saved)

    def test_external_change_is_not_overwritten(self):
        dataset = Dataset(self.root)
        self.path.write_bytes(self.original + b"\n")
        with self.assertRaises(ValueError):
            dataset.save(2, dict.fromkeys(PHASES, 1.0))
        self.assertEqual(self.path.read_bytes(), self.original + b"\n")

    def test_rejects_duplicate_episode_and_invalid_json(self):
        self.path.write_bytes(self.original + self.original)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            Dataset(self.root)
        self.path.write_text("{broken json}")
        with self.assertRaisesRegex(ValueError, "line 1"):
            Dataset(self.root)

    def test_invalid_timestamp_does_not_write(self):
        dataset = Dataset(self.root)
        for value in (-1, True, float("nan"), float("inf"), "1.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                dataset.save(2, {**dict.fromkeys(PHASES), PHASES[0]: value})
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_missing_one_video_keeps_episode_available(self):
        dataset = Dataset(self.root)
        dataset.videos[17].unlink()
        reloaded = Dataset(self.root)
        self.assertIn(17, reloaded.records)
        self.assertNotIn(17, reloaded.videos)

    def test_duplicate_video_is_rejected(self):
        folder = self.root / "videos/chunk-003" / CAMERA
        folder.mkdir(parents=True)
        (folder / "episode_000002.mp4").touch()
        with self.assertRaisesRegex(ValueError, "More than one video"):
            Dataset(self.root)

    def test_zero_template_and_format(self):
        self.assertEqual(phase_values(self.records[0]), dict.fromkeys(PHASES))
        record = deepcopy(self.records[0])
        record["phase_timestamps"]["idle_start"] = 1
        self.assertIsNone(phase_values(record)["approach_start"])
        self.assertEqual(format_time(5240), "05:240")
        self.assertEqual(format_time(125007), "125:007")


if __name__ == "__main__":
    unittest.main()
