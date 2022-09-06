#!/usr/bin/env python3

"""
@author: xi
@since: 2021-12-16
"""

import collections
from typing import Union, Tuple, Dict

import numpy as np

__all__ = [
    'APMeter',
    'MAPMeter'
]


class APMeter(object):

    def __init__(self, iou_threshold: float, data_format):
        self.iou_threshold = iou_threshold
        self.data_format = data_format
        self.target_dict = collections.defaultdict(list)
        self.output_list = []

        self._computed = False
        self._score = None
        self._matching = None
        self._precision = None
        self._recall = None
        self._ap = None

    def add_target(self, name: Union[str, int], bboxes: np.ndarray):
        """Add a target bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 5) in xywhc format.
        """
        self._computed = False
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 5
            for bbox in bboxes:
                self.target_dict[name].append(bbox)
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 5
            self.target_dict[name].append(bboxes)
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def add_output(self, name: Union[str, int], bboxes: np.ndarray):
        """Add an output bounding box to the meter.

        Args:
            name: Instance name, e.g., image path, image id.
            bboxes: A matrix/vector with shape (?, 6) in xywhcs format.
        """
        self._computed = False
        if len(bboxes.shape) == 2:
            assert bboxes.shape[-1] >= 6
            for bbox in bboxes:
                score = float(bbox[5])
                self.output_list.append([name, bbox, score])
        elif len(bboxes.shape) == 1:
            assert bboxes.shape[-1] >= 6
            score = float(bboxes[5])
            self.output_list.append([name, bboxes, score])
        else:
            raise RuntimeError(f'Invalid bboxes shape {bboxes.shape}')

    def compute(self):
        if self._computed:
            return

        if len(self.output_list) == 0 or len(self.target_dict) == 0:
            self._computed = True
            self._score = np.array([], dtype=np.float32)
            self._matching = np.array([], dtype=np.float32)
            self._precision = np.array([], dtype=np.float32)
            self._recall = np.array([], dtype=np.float32)
            self._ap = 0.0
            return

        self._score, self._matching = self._compute_matching_vector()
        self._precision, self._recall = self._compute_pr_curve(self._matching)
        ap = 0.0
        last_r = 0.0
        for i in range(len(self._recall)):
            r_i, p_i = self._recall[i], self._precision[i]
            ap += (r_i - last_r) * p_i
            last_r = r_i
        self._ap = ap
        self._computed = True
        return (
            self._score,
            self._matching,
            self._precision,
            self._recall,
            self._ap
        )

    def _compute_matching_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        self.output_list.sort(key=lambda _t: -_t[2])
        score = np.array([_t[2] for _t in self.output_list], dtype=np.ndarray)

        matching = np.zeros((len(self.output_list),), dtype=np.float32)
        matched_targets = set()
        for i, (name, bbox_output, _) in enumerate(self.output_list):
            for bbox_target in self.target_dict[name]:
                if id(bbox_target) in matched_targets:
                    continue
                if self._iou(bbox_output, bbox_target, self.data_format) >= self.iou_threshold:
                    matching[i] = 1
                    matched_targets.add(id(bbox_target))
                    break
        return score, matching

    def _compute_pr_curve(self, matching_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        tp = np.cumsum(matching_vec)  # num of the correctly predicted positives
        tp_fp = np.arange(1, len(matching_vec) + 1, dtype=np.float32)  # num of the predicted positives
        tp_fn = sum([len(_v) for _v in self.target_dict.values()])  # num of the true positives
        precision = tp / tp_fp
        recall = tp / tp_fn

        # smooth the pr curve
        for i in range(len(precision) - 1):
            precision[i] = precision[i + 1:].max()
        return precision, recall

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray, data_format, eps=1e-7) -> float:
        if data_format == 'xywh':
            b1_x, b1_y, b1_w, b1_h = bbox1[0], bbox1[1], bbox1[2] * 0.5, bbox1[3] * 0.5
            b2_x, b2_y, b2_w, b2_h = bbox2[0], bbox2[1], bbox2[2] * 0.5, bbox2[3] * 0.5
            b1_x1, b1_y1, b1_x2, b1_y2 = b1_x - b1_w, b1_y - b1_h, b1_x + b1_w, b1_y + b1_h
            b2_x1, b2_y1, b2_x2, b2_y2 = b2_x - b2_w, b2_y - b2_h, b2_x + b2_w, b2_y + b2_h
        elif data_format == 'xyxy':
            b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        else:
            raise RuntimeError(f'Unsupported data format "{data_format}".')
        w_inter = max(min(b1_x2, b2_x2) - max(b1_x1, b2_x1), 0)
        h_inter = max(min(b1_y2, b2_y2) - max(b1_y1, b2_y1), 0)
        area_inter = w_inter * h_inter

        area_a = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_b = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        area_union = area_a + area_b - area_inter

        iou = area_inter / (area_union + eps)
        return iou

    def score(self) -> np.ndarray:
        self.compute()
        return self._score

    def matching(self) -> np.ndarray:
        self.compute()
        return self._matching

    def precision(self) -> np.ndarray:
        self.compute()
        return self._precision

    def recall(self) -> np.ndarray:
        self.compute()
        return self._recall

    def ap(self) -> float:
        """Compute Average Precision (AP) score.

        Returns:
            A float number represents the AP score.
        """
        self.compute()
        return self._ap


class MAPMeter(object):

    def __init__(self, iou_threshold: float = 0.5, num_workers: int = 0.5, data_format='xywh'):
        """Mean Average Precision (MAP) Meter.

        Args:
            iou_threshold: A threshold used to determined whether the prediction bbox is correct.
        """
        self.ap_meters = collections.defaultdict(lambda: APMeter(iou_threshold, data_format))
        if num_workers is None:
            self.num_workers = 1
        elif isinstance(num_workers, float):
            assert num_workers <= 1
            import multiprocessing as mp
            self.num_workers = max(int(num_workers * mp.cpu_count()), 4)
        else:
            self.num_workers = num_workers

    def update(
            self,
            name: Union[str, int],
            output: np.ndarray,
            target: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add output (model prediction) and target (ground truth) for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            output: (n, 6) matrix in xywhcs format.
            target: (n, 5) matrix in xywhc format.
            label: The labels of the bboxes.
        """
        self.update_output(name, output, label)
        self.update_target(name, target, label)

    def update_output(
            self,
            name: Union[str, int],
            output: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add output bounding boxes for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            output: (n, 6) matrix in xywhcs format.
            label: The labels of the bboxes.
        """
        assert len(output.shape) == 2 and output.shape[1] >= 6
        if label is None:
            for bbox in output:
                self.ap_meters[int(bbox[4])].add_output(name, bbox)
        elif isinstance(label, np.ndarray):
            assert label.shape[0] == output.shape[0]
            for bbox, l in zip(output, label):
                self.ap_meters[int(l)].add_output(name, bbox)
        elif isinstance(label, int):
            for bbox in output:
                self.ap_meters[label].add_output(name, bbox)
        else:
            raise RuntimeError(f'Invalid label type {type(label)}')

    def update_target(
            self,
            name: Union[str, int],
            target: np.ndarray,
            label: Union[np.ndarray, int] = None
    ) -> None:
        """Add target bounding boxes for one single instance.

        Args:
            name: Instance name, e.g., image path, image id.
            target: (n, 5) matrix in xywhc format.
            label: The labels of the bboxes.
        """
        assert len(target.shape) == 2 and target.shape[1] >= 5
        if label is None:
            for bbox in target:
                self.ap_meters[int(bbox[4])].add_target(name, bbox)
        elif isinstance(label, np.ndarray):
            assert label.shape[0] == target.shape[0]
            for bbox, l in zip(target, label):
                self.ap_meters[int(l)].add_target(name, bbox)
        elif isinstance(label, int):
            for bbox in target:
                self.ap_meters[label].add_target(name, bbox)
        else:
            raise RuntimeError(f'Invalid label type {type(label)}')

    def m_ap(self) -> float:
        """Compute MAP score for all classes.

        Returns:
            A float number that represents the MAP score.
        """
        if len(self.ap_meters) == 0:
            return 0.0

        score = float(np.mean(list(self.ap().values())))
        return score

    def score(self) -> Dict[int, np.ndarray]:
        self._compute()
        return {c: m.score() for c, m in self.ap_meters.items()}

    def matching(self) -> Dict[int, np.ndarray]:
        self._compute()
        return {c: m.matching() for c, m in self.ap_meters.items()}

    def precision(self) -> Dict[int, np.ndarray]:
        self._compute()
        return {c: m.precision() for c, m in self.ap_meters.items()}

    def recall(self) -> Dict[int, np.ndarray]:
        self._compute()
        return {c: m.recall() for c, m in self.ap_meters.items()}

    def ap(self) -> Dict[int, np.ndarray]:
        self._compute()
        return {c: m.ap() for c, m in self.ap_meters.items()}

    def _compute(self):
        if self.num_workers <= 1:
            for meter in self.ap_meters.values():
                meter.compute()
        else:
            meter_list = self.ap_meters.values()

            import multiprocessing as mp
            with mp.Pool(self.num_workers) as pool:
                computed_list = [pool.apply_async(meter.compute) for meter in meter_list]
                computed_list = [computed.get() for computed in computed_list]

            for meter, computed in zip(meter_list, computed_list):
                (meter._score,
                 meter._matching,
                 meter._precision,
                 meter._recall,
                 meter._ap) = computed
                meter._computed = True
