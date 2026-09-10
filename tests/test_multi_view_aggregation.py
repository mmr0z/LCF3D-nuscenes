import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'mmdetection3d'))

from multi_view_inference import LateFusionMultiViewInferencer


class MultiViewAggregationTest(unittest.TestCase):

    @staticmethod
    def _inferencer():
        inferencer = object.__new__(LateFusionMultiViewInferencer)
        inferencer.use_label_fusion = False
        inferencer.use_score_fusion = False
        inferencer.keep_unmatched_3d = True
        inferencer.keep_oov_bboxes = False
        inferencer.oov_score_thr = 0.3
        return inferencer

    def test_identity_mode_keeps_each_lidar_detection_exactly_once(self):
        inferencer = self._inferencer()
        boxes = torch.arange(27, dtype=torch.float32).reshape(3, 9)
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 1, 2])

        output = inferencer._aggregate_lidar_detections(
            boxes,
            scores,
            labels,
            matched_source_mask=torch.tensor([True, True, False]),
            oov_mask=torch.tensor([False, False, True]),
            # Source 0 was matched in two different cameras.
            matched_indices_3d=[torch.tensor([0, 1]), torch.tensor([0])],
            matched_bboxes_2d=[
                torch.zeros((2, 4)),
                torch.zeros((1, 4)),
            ],
            matched_scores_2d=[torch.tensor([0.6, 0.7]), torch.tensor([0.9])],
            matched_labels_2d=[torch.tensor([0, 1]), torch.tensor([0])],
        )

        self.assertTrue(torch.equal(output[0], boxes))
        self.assertTrue(torch.equal(output[1], scores))
        self.assertTrue(torch.equal(output[2], labels))

    def test_semantic_fusion_uses_best_camera_match_per_source(self):
        inferencer = self._inferencer()
        inferencer.use_label_fusion = True
        inferencer.class_priors = torch.full((3,), 1 / 3)
        boxes = torch.arange(27, dtype=torch.float32).reshape(3, 9)
        scores = torch.tensor([0.9, 0.8, 0.7])
        labels = torch.tensor([0, 1, 2])

        output = inferencer._aggregate_lidar_detections(
            boxes,
            scores,
            labels,
            matched_source_mask=torch.tensor([True, True, False]),
            oov_mask=torch.tensor([False, False, True]),
            matched_indices_3d=[torch.tensor([0, 1]), torch.tensor([0])],
            matched_bboxes_2d=[
                torch.zeros((2, 4)),
                torch.zeros((1, 4)),
            ],
            matched_scores_2d=[torch.tensor([0.6, 0.7]), torch.tensor([0.9])],
            matched_labels_2d=[torch.tensor([1, 1]), torch.tensor([2])],
        )

        self.assertTrue(torch.equal(output[0], boxes))
        self.assertTrue(torch.equal(output[1], scores))
        self.assertTrue(torch.equal(output[2], torch.tensor([2, 1, 2])))

    def test_empty_recovery_is_an_exact_noop(self):
        inferencer = self._inferencer()
        boxes = torch.arange(18, dtype=torch.float32).reshape(2, 9)
        scores = torch.tensor([0.9, 0.8])
        labels = torch.tensor([0, 1])

        output = inferencer._merge_recovered_detections(
            boxes,
            scores,
            labels,
            torch.empty((0, 7)),
            torch.empty((0,)),
            torch.empty((0,), dtype=torch.long),
        )

        self.assertIs(output[0], boxes)
        self.assertIs(output[1], scores)
        self.assertIs(output[2], labels)


if __name__ == '__main__':
    unittest.main()
