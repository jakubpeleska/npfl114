from typing import Tuple, List, Callable

import tensorflow as tf


class AnchorsFactory:
    SCALES = [1., 2. ** (1. / 3.), 2. ** (2. / 3.)]
    RATIOS = [1. / 2., 1., 2.]

    @staticmethod
    def get_default_areas(l_low: int, l_high: int) -> List[int]:
        areas = [2 ** (2 + i) for i in range(l_low, l_high + 1)]
        return [x ** 2 for x in areas]

    @staticmethod
    def get_default_strides(l_low: int, l_high: int) -> List[int]:
        return [2 ** i for i in range(l_low, l_high + 1)]

    def __init__(self,
                 levels: Tuple[int, int],
                 scales: List[float] = None,
                 ratios: List[float] = None,
                 stride_provider: Callable[[Tuple[int, int]], List[int]] = None,
                 area_provider: Callable[[Tuple[int, int]], List[int]] = None):
        l_low, l_high = levels
        self.levels = list(range(l_low, l_high + 1))

        if stride_provider is None:
            stride_provider = AnchorsFactory.get_default_strides
        if area_provider is None:
            area_provider = AnchorsFactory.get_default_areas

        self.strides = stride_provider(l_low, l_high)
        self.areas = area_provider(l_low, l_high)
        self.scales = scales if scales is not None else AnchorsFactory.SCALES
        self.ratios = ratios if ratios is not None else AnchorsFactory.RATIOS
        assert len(self.strides) == len(self.areas)

    @property
    def anchors_per_unit(self) -> int:
        return len(self.scales) * len(self.ratios)

    def get_grid_shape(self, level: int, height: int, width: int):
        y_repeats = tf.math.ceil(height / (2 ** level))
        x_repeats = tf.math.ceil(width / (2 ** level))
        return y_repeats, x_repeats

    def build(self, height: int, width: int) -> tf.Tensor:
        anchors = []

        for level, stride, area in zip(self.levels, self.strides, self.areas):
            y_repeats, x_repeats = self.get_grid_shape(level, height, width)
            ry = (tf.range(y_repeats, dtype=tf.float32) + 0.5) * stride
            rx = (tf.range(x_repeats, dtype=tf.float32) + 0.5) * stride

            centers = tf.stack(tf.meshgrid(ry, rx, indexing='ij'), axis=-1)
            centers = tf.reshape(centers, (-1, 2))
            anchor_configs = tf.tile(centers, (1, 2))

            anchors_relative = []

            for scale in self.scales:
                for ratio in self.ratios:
                    h = tf.math.sqrt(area / ratio)
                    w = area / h

                    h = h * scale
                    w = w * scale

                    anchors_relative.append(
                        tf.convert_to_tensor([
                            -h / 2.,  # top
                            -w / 2.,  # left
                            h / 2.,  # bottom
                            w / 2.,  # right
                        ])
                    )
            anchors_relative = tf.stack(anchors_relative, axis=0)
            anchors.append(tf.reshape((anchor_configs[:, None] + anchors_relative[None]), (-1, 4)))

        return tf.concat(anchors, axis=0)
