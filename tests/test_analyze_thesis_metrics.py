import json
import tempfile
import unittest
from pathlib import Path

from src.analyze_thesis_metrics import parse_run, write_outputs


class ThesisMetricsTest(unittest.TestCase):

    def test_parses_official_nuscenes_metrics(self):
        payload = {
            "mean_ap": 0.31,
            "nd_score": 0.44,
            "tp_errors": {"trans_err": 0.5},
            "mean_dist_aps": {"car": 0.7},
            "label_aps": {"car": {"0.5": 0.6, "1.0": 0.8}},
            "label_tp_errors": {"car": {"trans_err": 0.2}},
            "lcf3d_runtime": {
                "mean_seconds_per_sample": 0.25,
                "samples_per_second": 4.0,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            summary, rows = parse_run(path, "fusion")

        self.assertEqual(summary["protocol"], "nuScenes-3D")
        self.assertEqual(summary["AP"], 0.31)
        self.assertEqual(summary["latency_ms"], 250.0)
        self.assertEqual(rows[0]["AP_0.5m"], 0.6)
        self.assertEqual(rows[0]["ATE"], 0.2)

    def test_merges_coco_jsonl_and_classwise_log_table(self):
        metric = {
            "coco/car_precision": 0.3,
            "coco/bbox_mAP": 0.2,
            "coco/bbox_mAP_50": 0.4,
            "coco/bbox_AR@100": 0.5,
            "time": 0.05,
            "step": 12,
        }
        log = """
| category | mAP | mAP_50 | mAP_75 | mAP_s | mAP_m | mAP_l |
| car | 0.3 | 0.6 | 0.2 | 0.1 | 0.3 | 0.5 |
"""
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            vis_dir = run_dir / "vis_data"
            vis_dir.mkdir()
            (vis_dir / "scalars.json").write_text(
                json.dumps({"loss": 1.0}) + "\n" + json.dumps(metric) + "\n",
                encoding="utf-8",
            )
            (run_dir / "test.log").write_text(log, encoding="utf-8")
            summary, rows = parse_run(run_dir, "camera")

        self.assertEqual(summary["protocol"], "COCO-2D")
        self.assertEqual(summary["AR100"], 0.5)
        self.assertAlmostEqual(summary["throughput_per_s"], 20.0)
        self.assertEqual(rows[0]["AP50"], 0.6)
        self.assertEqual(rows[0]["AP_large"], 0.5)

    def test_writes_machine_and_thesis_outputs(self):
        summary = {"run": "example", "protocol": "COCO-2D", "AP": 0.1}
        row = {"run": "example", "protocol": "COCO-2D", "class": "car", "AP": 0.2}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_outputs(output_dir, [summary], [row])
            self.assertTrue((output_dir / "summary.csv").is_file())
            self.assertTrue((output_dir / "per_class.csv").is_file())
            report = (output_dir / "thesis_report.md").read_text(encoding="utf-8")
            self.assertIn("example", report)
            self.assertIn("COCO AP", report)


if __name__ == "__main__":
    unittest.main()
