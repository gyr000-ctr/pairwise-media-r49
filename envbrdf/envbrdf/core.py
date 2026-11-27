"""envbrdf 核心 API：GGX 环境 BRDF 查表（仅标准库）。

表布局
------
``table[row][col]``：

* 行（row / y 轴）对应粗糙度 roughness，在 ``[0, 1]`` 上均匀采样；
* 列（col / x 轴）对应 ``n_dot_v``，在 ``[0, 1]`` 上均匀采样。

``lookup`` 在归一化坐标上做双线性插值。插值在采样节点处逐位精确
（返回表中存储的原值），因此由同一解析 GGX 近似式采样得到的表在节点处
与解析值的误差为 0（远小于 1e-9）。

边界语义
--------
* 两个坐标都落在定义域外（例如同时 > 1）时，饱和 clamp 到表的角点，
  不会造成下标越界；
* 只有单个坐标越界时视为非法输入并抛出可区分的错误：
  - 粗糙度越界 -> :class:`RoughnessOutOfRangeError`
  - ``n_dot_v`` 越界 -> :class:`NdVOutOfRangeError`
* 非法表 -> :class:`InvalidTableError`

以上错误类型都继承 ``ValueError``，既可用 ``except ValueError`` 捕获，
也可通过具体类型区分“非法表”和“越界粗糙度”。

能量与单调性
------------
查表结果是四个表项以非负权重（和为 1）的凸组合；当所用表项都在
``[0, 1]`` 内时结果必在 ``[0, 1]`` 内（单词查表能量 ≤ 1）。
:func:`ggx_half_width` 给出解析 GGX 单瓣（NDF）的半高全宽：
``FWHM(roughness) = 2 * atan(sqrt(sqrt(2) - 1) * roughness**2)``，
随粗糙度单调不减，即粗糙度越大高光越宽。
"""

import math

__all__ = [
    "EnvBrdf",
    "EnvBrdfError",
    "InvalidTableError",
    "RoughnessOutOfRangeError",
    "NdVOutOfRangeError",
    "ggx_half_width",
]


class EnvBrdfError(ValueError):
    """envbrdf 的基类错误（继承 ``ValueError`` 以兼容调用方）。"""


class InvalidTableError(EnvBrdfError):
    """表非法：为空、非矩形、含非数值 / 非有限值等。"""


class RoughnessOutOfRangeError(EnvBrdfError):
    """粗糙度越界（负值或 NaN）。"""


class NdVOutOfRangeError(EnvBrdfError):
    """``n_dot_v`` 越界（非数值、NaN 或不在合法区间）。"""


# sqrt(sqrt(2) - 1)：GGX 半高角 theta 满足 tan(theta) = k * alpha 时的系数。
_HALF_MAX_K = math.sqrt(math.sqrt(2.0) - 1.0)


def ggx_half_width(roughness):
    """解析 GGX 单瓣半高全宽（弧度），关于粗糙度单调不减。

    GGX 微表面分布（``alpha = roughness**2``）在 ``theta = 0`` 处取峰值；
    令 ``D(theta) = D(0) / 2`` 可解得 ``tan(theta) = sqrt(sqrt(2) - 1) * alpha``，
    因此 FWHM = ``2 * theta``。``roughness = 0`` 时退化为 0。
    """
    alpha = roughness * roughness
    return 2.0 * math.atan(_HALF_MAX_K * alpha)


def _is_real(value):
    """``value`` 是否为非 bool 的实数（int / float）。"""
    kind = type(value)
    return kind is float or kind is int


def _normalize_table(table):
    """校验并把表转换为 ``(rows, nrows, ncols)`` 的浮点列表。"""
    if isinstance(table, (str, bytes)):
        raise InvalidTableError("invalid table: expected a 2-D sequence, got text")
    try:
        raw_rows = list(table)
    except TypeError:
        raise InvalidTableError("invalid table: expected a 2-D sequence") from None
    if not raw_rows:
        raise InvalidTableError("invalid table: no rows (empty table)")

    rows = []
    width = None
    for row_index, raw_row in enumerate(raw_rows):
        if isinstance(raw_row, (str, bytes)):
            raise InvalidTableError(
                "invalid table: row {} is text, expected a sequence of numbers".format(
                    row_index
                )
            )
        try:
            values = list(raw_row)
        except TypeError:
            raise InvalidTableError(
                "invalid table: row {} is not a sequence".format(row_index)
            ) from None
        if not values:
            raise InvalidTableError(
                "invalid table: row {} is empty".format(row_index)
            )
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise InvalidTableError(
                "invalid table: row {} has {} columns, expected {}".format(
                    row_index, len(values), width
                )
            )
        converted = []
        for column_index, value in enumerate(values):
            if not _is_real(value):
                raise InvalidTableError(
                    "invalid table: cell ({}, {}) is not a number".format(
                        row_index, column_index
                    )
                )
            number = float(value)
            if not math.isfinite(number):
                raise InvalidTableError(
                    "invalid table: cell ({}, {}) is not finite".format(
                        row_index, column_index
                    )
                )
            converted.append(number)
        rows.append(converted)

    return rows, len(rows), width


class EnvBrdf:
    """从二维表构造的环境 BRDF，支持双线性查表与解析半高宽。"""

    __slots__ = ("_rows", "_nrows", "_ncols")

    def __init__(self, table):
        rows, nrows, ncols = _normalize_table(table)
        # 复制到私有存储，调用方后续修改原表不会影响结果。
        self._rows = tuple(tuple(row) for row in rows)
        self._nrows = nrows
        self._ncols = ncols

    def lookup(self, n_dot_v, roughness):
        """双线性查表。

        参数按 ``(n_dot_v, roughness)`` 传入。两个坐标都越界时饱和 clamp 到
        角点；仅一个坐标越界时抛出可区分的错误。
        """
        if not _is_real(n_dot_v):
            raise NdVOutOfRangeError("n_dot_v out of range: expected a real number")
        if not _is_real(roughness):
            raise RoughnessOutOfRangeError(
                "roughness out of range: expected a real number"
            )
        if n_dot_v != n_dot_v:
            raise NdVOutOfRangeError("n_dot_v out of range: NaN")
        if roughness != roughness:
            raise RoughnessOutOfRangeError("roughness out of range: NaN")

        x_out = n_dot_v < 0.0 or n_dot_v > 1.0
        y_out = roughness < 0.0 or roughness > 1.0
        if y_out:
            if not x_out:
                raise RoughnessOutOfRangeError(
                    "roughness out of range: {} not in [0, 1]".format(roughness)
                )
            # 两个坐标同时越界：饱和到表角点，clamp 不会越界。
            x = 1.0 if n_dot_v > 1.0 else 0.0
            y = 1.0 if roughness > 1.0 else 0.0
        elif x_out:
            raise NdVOutOfRangeError(
                "n_dot_v out of range: {} not in [0, 1]".format(n_dot_v)
            )
        else:
            x = float(n_dot_v)
            y = float(roughness)

        rows = self._rows
        nrows = self._nrows
        ncols = self._ncols

        # 列方向 = n_dot_v (x)，行方向 = roughness (y)。
        fx = x * (ncols - 1)
        ix = int(fx)
        if ix >= ncols - 1:
            ix = ncols - 1
            ix1 = ix
            tx = 0.0
        else:
            ix1 = ix + 1
            tx = fx - ix

        fy = y * (nrows - 1)
        iy = int(fy)
        if iy >= nrows - 1:
            iy = nrows - 1
            iy1 = iy
            ty = 0.0
        else:
            iy1 = iy + 1
            ty = fy - iy

        row0 = rows[iy]
        row1 = rows[iy1]
        v00 = row0[ix]
        v10 = row0[ix1]
        v01 = row1[ix]
        v11 = row1[ix1]

        # 标准双线性插值：节点处（tx/ty = 0）逐位返回原值。
        low = (v00 * (1.0 - tx) + v10 * tx) * (1.0 - ty)
        high = (v01 * (1.0 - tx) + v11 * tx) * ty
        value = low + high

        # 当四角都在 [0, 1] 内时，凸组合在数学上已落在 [0, 1]；这里只消除
        # 浮点舍入带来的极微小越界，保证能量守恒表的结果严格 ≤ 1。
        # 若表本身含 [0, 1] 之外的值，则保留精确插值，不做截断。
        if (
            0.0 <= v00 <= 1.0
            and 0.0 <= v10 <= 1.0
            and 0.0 <= v01 <= 1.0
            and 0.0 <= v11 <= 1.0
        ):
            if value < 0.0:
                value = 0.0
            elif value > 1.0:
                value = 1.0
        return value

    def fwhm(self, roughness):
        """给定粗糙度下的高光半高全宽（弧度），随粗糙度单调不减。"""
        if not _is_real(roughness):
            raise RoughnessOutOfRangeError(
                "roughness out of range: expected a real number"
            )
        if roughness != roughness:
            raise RoughnessOutOfRangeError("roughness out of range: NaN")
        if roughness < 0.0:
            raise RoughnessOutOfRangeError(
                "roughness out of range: {} < 0".format(roughness)
            )
        if roughness > 1.0:
            roughness = 1.0
        return ggx_half_width(float(roughness))
