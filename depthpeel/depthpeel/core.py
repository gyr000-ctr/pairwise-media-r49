"""depthpeel 核心 API：深度剥离与层覆盖面积统计（仅标准库）。

模型说明
--------
深度剥离把输入的一组深度样本按由近到远切分成若干层。本实现约定每个样本
代表一个覆盖面积为 ``1.0`` 的三角形，因此某一层的覆盖面积就是落入该层的
样本数（以 float 返回），未剥离前的整体覆盖面积（"首层面积"）为样本总数。
所有层面积之和始终等于该整体面积：超过层数上限或触发面积阈值提前终止时，
更深层的样本会并入最后一层，不会被丢弃。

层深取每簇的最小值；相邻层深严格递增且间距大于 ``1e-12``，间距不超过该
容差的样本视为同一深度的重复样本。

异常
----
``InvalidLayerCount`` 与 ``InvalidDepth`` 都是 ``ValueError`` 的子类，但类型
和消息不同，可区分"非法层数"与"负深度/非法深度"。
"""

import math
import numbers

__all__ = [
    "DepthPeel",
    "DepthPeelError",
    "InvalidLayerCount",
    "InvalidDepth",
    "InvalidAreaThreshold",
]

#: 层深合并容差：间距 <= 该值视为同一层。
LAYER_TOLERANCE = 1e-12


class DepthPeelError(ValueError):
    """depthpeel 参数错误的基类。"""


class InvalidLayerCount(DepthPeelError):
    """层数非法（必须为正整数）。"""


class InvalidDepth(DepthPeelError):
    """深度样本非法（必须为有限、非负的实数）。"""


class InvalidAreaThreshold(DepthPeelError):
    """面积阈值非法（必须为 [0, 1] 内的有限实数）。"""


def _validate_layer_count(layers):
    """校验层数，返回 int。"""
    if isinstance(layers, bool) or not isinstance(layers, numbers.Integral):
        raise InvalidLayerCount(
            "invalid layer count: expected a positive integer, got %r" % (layers,)
        )
    layers = int(layers)
    if layers <= 0:
        raise InvalidLayerCount(
            "invalid layer count: expected a positive integer, got %r" % (layers,)
        )
    return layers


def _validate_area_threshold(threshold):
    """校验面积阈值，返回 float。"""
    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise InvalidAreaThreshold(
            "invalid area threshold: expected a real number in [0, 1], got %r"
            % (threshold,)
        )
    value = float(threshold)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise InvalidAreaThreshold(
            "invalid area threshold: expected a finite number in [0, 1], got %r"
            % (threshold,)
        )
    return value


def _validate_depths(depths):
    """校验深度样本并返回新的 float 列表（不修改输入）。"""
    try:
        raw = list(depths)
    except TypeError:
        raise InvalidDepth(
            "invalid depth: expected an iterable of numbers, got %r" % (depths,)
        ) from None

    values = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise InvalidDepth(
                "invalid depth: expected a real number, got %r" % (value,)
            )
        number = float(value)
        if not math.isfinite(number):
            raise InvalidDepth(
                "invalid depth: expected a finite number, got %r" % (value,)
            )
        if number < 0.0:
            raise InvalidDepth(
                "invalid depth: expected a non-negative number, got %r" % (value,)
            )
        values.append(number)
    return values


class DepthPeel:
    """按深度由近到远剥离，并统计每层覆盖面积。

    参数
    ----
    layers:
        最多保留的层数（正整数）。
    area_threshold:
        面积阈值。若剩余未剥离样本的覆盖面积占比不超过该值，则不再新增层，
        剩余样本并入当前最后一层。默认 ``0.0`` 表示只受 ``layers`` 限制。
    """

    def __init__(self, layers, area_threshold=0.0):
        self._layers = _validate_layer_count(layers)
        self._area_threshold = _validate_area_threshold(area_threshold)
        self._depths = []
        self._counts = []
        self._total = 0

    def peel(self, depths):
        """剥离深度样本，返回由近到远的层深列表（float）。

        输入不会被修改；相邻返回层深的间距严格大于 ``1e-12``；最多返回
        ``layers`` 层。空输入返回空列表。
        """
        values = _validate_depths(depths)
        total = len(values)

        if total == 0:
            self._depths = []
            self._counts = []
            self._total = 0
            return []

        values.sort()

        # 排序后按容差切分：每簇保留最小值作为该层深度并统计覆盖样本数。
        reps = []
        counts = []
        for value in values:
            if reps and value - reps[-1] <= LAYER_TOLERANCE:
                counts[-1] += 1
            else:
                reps.append(value)
                counts.append(1)

        # 面积阈值自适应终止：剩余面积占比 <= 阈值时，不再新增层。
        limit = len(reps)
        if self._area_threshold > 0.0:
            remaining = total
            for index, count in enumerate(counts):
                remaining -= count
                if remaining / total <= self._area_threshold:
                    limit = index + 1
                    break

        # 层数上限同样截断；两种截断都必须保证面积守恒。
        cut = min(self._layers, limit, len(reps))
        if cut < len(reps):
            counts[cut - 1] += sum(counts[cut:])
            del counts[cut:]
            del reps[cut:]

        self._depths = reps
        self._counts = counts
        self._total = total
        return list(reps)

    def count(self):
        """最近一次剥离得到的层数（int；尚未剥离时为 0）。"""
        return len(self._depths)

    def depths(self):
        """最近一次剥离的层深副本。"""
        return list(self._depths)

    def areas(self):
        """最近一次剥离的每层覆盖面积（每个样本面积记为 1.0）。

        返回列表长度等于 :meth:`count`；各层面积之和等于 :meth:`base_area`。
        """
        return [float(count) for count in self._counts]

    def relative_areas(self):
        """每层覆盖面积占整体面积的比例，之和为 1。"""
        if self._total == 0:
            return []
        return [count / self._total for count in self._counts]

    def base_area(self):
        """未剥离前的整体覆盖面积（即最近一次输入的样本总数）。"""
        return float(self._total)

    def total_area(self):
        """各层覆盖面积之和（与 :meth:`base_area` 相等）。"""
        return float(sum(self._counts))
