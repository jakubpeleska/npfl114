import os
import sys
from typing import Any, Dict, List, Tuple, Sequence, TextIO, Union
import urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import numpy.typing as npt
import tensorflow as tf


class SVHN:
    LABELS: int = 10

    # Type alias for a bounding box -- a list of floats.
    BBox = List[float]

    # The indices of the bounding box coordinates.
    TOP: int = 0
    LEFT: int = 1
    BOTTOM: int = 2
    RIGHT: int = 3

    _URL: str = "https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/datasets/"

    @staticmethod
    def parse(example: tf.Tensor) -> Dict[str, tf.Tensor]:
        example = tf.io.parse_single_example(example, {
            "image": tf.io.FixedLenFeature([], tf.string),
            "classes": tf.io.VarLenFeature(tf.int64),
            "bboxes": tf.io.VarLenFeature(tf.int64)})
        example["image"] = tf.image.decode_png(example["image"], channels=3)
        example["classes"] = tf.sparse.to_dense(example["classes"])
        example["bboxes"] = tf.reshape(tf.cast(tf.sparse.to_dense(example["bboxes"]), tf.float32), [-1, 4])
        return example

    def __init__(self) -> None:
        for dataset, size in [("train", 10_000), ("dev", 1_267), ("test", 4_535)]:
            path = "svhn.{}.tfrecord".format(dataset)
            if not os.path.exists(path):
                print("Downloading file {}...".format(path), file=sys.stderr)
                urllib.request.urlretrieve("{}/{}".format(self._URL, path), filename="{}.tmp".format(path))
                os.rename("{}.tmp".format(path), path)

            setattr(self, dataset,
                    tf.data.TFRecordDataset(path).map(SVHN.parse).apply(tf.data.experimental.assert_cardinality(size)))

    train: tf.data.Dataset
    dev: tf.data.Dataset
    test: tf.data.Dataset

    # Evaluation infrastructure.
    @staticmethod
    def evaluate(
        gold_dataset: tf.data.Dataset, predictions: Sequence[Tuple[List[int], List[BBox]]], iou_threshold: float = 0.5,
    ) -> float:
        def bbox_iou(x: SVHN.BBox, y: SVHN.BBox) -> float:
            def area(bbox: SVHN.BBox) -> float:
                return max(bbox[SVHN.BOTTOM] - bbox[SVHN.TOP], 0) * max(bbox[SVHN.RIGHT] - bbox[SVHN.LEFT], 0)
            intersection = [max(x[SVHN.TOP], y[SVHN.TOP]), max(x[SVHN.LEFT], y[SVHN.LEFT]),
                            min(x[SVHN.BOTTOM], y[SVHN.BOTTOM]), min(x[SVHN.RIGHT], y[SVHN.RIGHT])]
            x_area, y_area, intersection_area = area(x), area(y), area(intersection)
            return intersection_area / (x_area + y_area - intersection_area)

        gold = [(np.array(example["classes"]), np.array(example["bboxes"])) for example in gold_dataset]

        if len(predictions) != len(gold):
            raise RuntimeError("The predictions are of different size than gold data: {} vs {}".format(
                len(predictions), len(gold)))

        correct = 0
        for (gold_classes, gold_bboxes), (prediction_classes, prediction_bboxes) in zip(gold, predictions):
            if len(gold_classes) != len(prediction_classes):
                continue

            used = [False] * len(gold_classes)
            for cls, bbox in zip(prediction_classes, prediction_bboxes):
                best = None
                for i in range(len(gold_classes)):
                    if used[i] or gold_classes[i] != cls:
                        continue
                    iou = bbox_iou(bbox, gold_bboxes[i])
                    if iou >= iou_threshold and (best is None or iou > best_iou):
                        best, best_iou = i, iou
                if best is None:
                    break
                used[best] = True
            correct += all(used)

        return 100 * correct / len(gold)

    @staticmethod
    def evaluate_file(gold_dataset: tf.data.Dataset, predictions_file: TextIO) -> float:
        predictions = []
        for line_n, line in enumerate(predictions_file, 1):
            values = line.split()
            if len(values) % 5:
                raise RuntimeError("Each prediction must contain multiple of 5 numbers, found {} on line {}".format(
                    len(values), line_n))

            predictions.append(([], []))
            for i in range(0, len(values), 5):
                predictions[-1][0].append(int(values[i]))
                predictions[-1][1].append([float(value) for value in values[i + 1:i + 5]])

        return SVHN.evaluate(gold_dataset, predictions)

    # Visualization infrastructure.
    def visualize(image: npt.ArrayLike, labels: List[Any], bboxes: List[BBox], show: bool) -> Union[None, bytes]:
        """Visualize the given image plus recognized objects.

        Arguments:
        - `image` is NumPy/TensorFlow input image with pixels in range [0-255];
        - `labels` is a list of labels to be shown using the `str` method;
        - `bboxes` is a list of `BBox`es (fourtuples TOP, LEFT, BOTTOM, RIGHT);
        - `show` controls whether to show the figure or return a PNG file:
          - if `True`, the figure is shown using `plt.show()`;
          - if `False`, the figure is saved as a PNG file and returned as `bytes`.
            It can then be saved to a file, or converted to a `tf.Tensor` using the
            `tf.image.decode_png` and added to Tensorboard via `tf.summary.image`.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(4, 4))
        plt.axis("off")
        plt.imshow(np.asarray(image, np.uint8))
        for label, (top, left, bottom, right) in zip(labels, bboxes):
            plt.gca().add_patch(plt.Rectangle(
                [left, top], right - left, bottom - top, fill=False, edgecolor=[1, 0, 1], linewidth=2))
            plt.gca().text(left, top, str(label), bbox={"facecolor": [1, 0, 1], "alpha": 0.5},
                           clip_box=plt.gca().clipbox, clip_on=True, ha="left", va="top")

        if show:
            plt.show()
        else:
            import io
            image = io.BytesIO()
            plt.savefig(image, format="png", dpi=125, bbox_inches="tight", pad_inches=0)
            plt.close()
            return image.getvalue()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", default=None, type=str, help="Prediction file to evaluate")
    parser.add_argument("--visualize", default=None, type=str, help="Prediction file to visualize")
    parser.add_argument("--dataset", default="dev", type=str, help="Gold dataset to evaluate")
    args = parser.parse_args()

    if args.evaluate:
        with open(args.evaluate, "r", encoding="utf-8-sig") as predictions_file:
            accuracy = SVHN.evaluate_file(getattr(SVHN(), args.dataset), predictions_file)
        print("SVHN accuracy: {:.2f}%".format(accuracy))

    if args.visualize:
        with open(args.visualize, "r", encoding="utf-8-sig") as predictions_file:
            for line, example in zip(predictions_file, getattr(SVHN(), args.dataset)):
                values = line.split()
                classes, bboxes = [], []
                for i in range(0, len(values), 5):
                    classes.append(values[i])
                    bboxes.append([float(value) for value in values[i + 1:i + 5]])
                SVHN.visualize(example["image"], classes, bboxes, show=True)
