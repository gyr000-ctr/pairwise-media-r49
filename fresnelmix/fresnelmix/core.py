"""fresnelmix 核心 API（仅标准库）。

菲涅尔反射：ior 为相对折射率 eta = n_transmitted / n_incident。

- eta > 1：从光疏进入光密介质（如空气 -> 玻璃），无全内反射；
- 0 < eta < 1：从光密进入光疏介质，入射角超过临界角时发生全内
  反射（TIR），s/p 分量反射率均为 1；
- 复数 eta = n + i*k（n > 0, k >= 0）：导体/吸收介质。

s/p 偏振分量由精确菲涅尔公式给出；非偏振反射率为两者平均。
Schlick 近似仅作为加速分支：仅在构造时扫描验证过与精确式偏差
< 5e-4（留 2 倍余量于 1e-3 约束）的入射角区间内启用，其余区间
回退到精确式。
"""

import cmath
import math

__all__ = [
    "FresnelMix",
    "FresnelError",
    "InvalidIORError",
    "CosineOutOfRangeError",
]

# Schlick 加速分支的验证容差（题面约束 1e-3，取一半作安全余量）
_SCHLICK_TOL = 5e-4
# 构造时扫描的余弦步长
_SCAN_STEP = 5e-4


class FresnelError(ValueError):
    """fresnelmix 参数错误基类。"""


class InvalidIORError(FresnelError):
    """非法折射率（eta <= 0、eta == 1、k < 0、NaN/Inf 或类型不支持）。"""


class CosineOutOfRangeError(FresnelError):
    """入射角余弦越界（不在 [0, 1] 或为 NaN）。"""


def _validate_cos(cos_theta):
    if isinstance(cos_theta, bool) or not isinstance(cos_theta, (int, float)):
        raise CosineOutOfRangeError(
            "cos_theta 必须是实数，得到 %r" % (cos_theta,))
    c = float(cos_theta)
    if math.isnan(c) or c < 0.0 or c > 1.0:
        raise CosineOutOfRangeError(
            "cos_theta 必须在 [0, 1] 内，得到 %r" % (cos_theta,))
    return c


class FresnelMix:
    """菲涅尔反射率与漫反射/镜面混合。

    ior: 相对折射率 eta = n_t / n_i（正实数且 != 1），或复数
    n + i*k（n > 0, k >= 0）表示吸收介质。
    """

    __slots__ = ("_n", "_k", "_is_conductor", "_f0", "_schlick_cmin")

    def __init__(self, ior):
        if isinstance(ior, bool) or not isinstance(ior, (int, float, complex)):
            raise InvalidIORError("非法折射率类型: %r" % (ior,))
        if isinstance(ior, complex):
            n, k = float(ior.real), float(ior.imag)
            if k < 0.0:
                raise InvalidIORError("消光系数 k 必须 >= 0: %r" % (ior,))
            if n <= 0.0:
                raise InvalidIORError("折射率实部必须 > 0: %r" % (ior,))
        else:
            n, k = float(ior), 0.0
            if n == 1.0:
                raise InvalidIORError("相对折射率不能等于 1（无界面）: %r" % (ior,))
            if n <= 0.0:
                raise InvalidIORError("折射率必须 > 0: %r" % (ior,))
        if not (math.isfinite(n) and math.isfinite(k)):
            raise InvalidIORError("折射率必须为有限值: %r" % (ior,))
        self._n = n
        self._k = k
        self._is_conductor = k > 0.0
        if self._is_conductor:
            r = complex(n, k)
            f0c = (r - 1.0) / (r + 1.0)
            self._f0 = f0c.real * f0c.real + f0c.imag * f0c.imag
        else:
            t = (n - 1.0) / (n + 1.0)
            self._f0 = t * t
        self._schlick_cmin = self._calibrate_schlick()

    @property
    def ior(self):
        """相对折射率（导体返回复数）。"""
        if self._is_conductor:
            return complex(self._n, self._k)
        return self._n

    def f0(self):
        """法向反射率 F0 = ((n-1)/(n+1))^2（导体为模方）。"""
        return self._f0

    # ---- 精确菲涅尔 ----

    def reflectance_sp(self, cos_theta):
        """精确菲涅尔 s/p 偏振反射率，返回 (Rs, Rp)。

        全内反射临界角之外（含临界角）两者均为 1.0。
        """
        c = _validate_cos(cos_theta)
        if self._is_conductor:
            return self._sp_conductor(c)
        return self._sp_dielectric(c)

    def _sp_dielectric(self, cos_i):
        n = self._n
        sin2_t = (1.0 - cos_i * cos_i) / (n * n)
        if sin2_t >= 1.0:
            # 全内反射（仅 eta < 1 时可达）
            return 1.0, 1.0
        cos_t = math.sqrt(1.0 - sin2_t)
        n_cos_i = n * cos_i
        n_cos_t = n * cos_t
        rs = (cos_i - n_cos_t) / (cos_i + n_cos_t)
        rp = (n_cos_i - cos_t) / (n_cos_i + cos_t)
        return rs * rs, rp * rp

    def _sp_conductor(self, cos_i):
        eta = complex(self._n, self._k)
        sin2_i = 1.0 - cos_i * cos_i
        cos_t = cmath.sqrt(1.0 - sin2_i / (eta * eta))
        rs = (cos_i - eta * cos_t) / (cos_i + eta * cos_t)
        rp = (eta * cos_i - cos_t) / (eta * cos_i + cos_t)
        return abs(rs) ** 2, abs(rp) ** 2

    def reflectance(self, cos_theta):
        """非偏振反射率（s/p 平均），精确菲涅尔公式。"""
        rs, rp = self.reflectance_sp(cos_theta)
        return 0.5 * (rs + rp)

    # ---- Schlick 加速分支 ----

    def _schlick_raw(self, c):
        t = 1.0 - c
        t5 = t * t
        t5 = t5 * t5 * t
        return self._f0 + (1.0 - self._f0) * t5

    def _calibrate_schlick(self):
        """扫描确定 Schlick 与精确式偏差 < _SCHLICK_TOL 的最小余弦。

        返回阈值 cmin：c >= cmin 时 Schlick 快速分支可安全使用。
        """
        cmin = 1.0
        steps = int(1.0 / _SCAN_STEP)
        for i in range(steps, -1, -1):
            c = i * _SCAN_STEP
            if abs(self._schlick_raw(c) - self.reflectance(c)) >= _SCHLICK_TOL:
                # 回退一个步长作为安全边界
                cmin = min(1.0, (i + 1) * _SCAN_STEP)
                break
        else:
            cmin = 0.0
        return cmin

    def schlick(self, cos_theta):
        """Schlick 近似（加速分支），与精确式偏差 <= 1e-3。

        在构造时验证过的安全区间（近法向）内使用 Schlick 五次幂
        快速公式；区间之外回退到精确菲涅尔式，保证任意入射角下
        返回值与精确式的偏差都不超过 1e-3。全内反射区域返回 1.0。
        """
        c = _validate_cos(cos_theta)
        if c >= self._schlick_cmin:
            if not self._is_conductor and self._n < 1.0:
                sin2_t = (1.0 - c * c) / (self._n * self._n)
                if sin2_t >= 1.0:
                    return 1.0
            return self._schlick_raw(c)
        return self.reflectance(c)

    # ---- 混合 ----

    def mix(self, diffuse, specular, cos_theta):
        """按菲涅尔权重混合：diffuse*(1-F) + specular*F。

        权重 (1-F) 与 F 之和恒为 1。输入为标量或等长序列；标量
        输入按三分量（RGB）处理并返回三元组，序列输入返回同长度
        元组。
        """
        f = self.reflectance(cos_theta)
        w_d = 1.0 - f
        w_s = f
        if isinstance(diffuse, (int, float)) and isinstance(specular, (int, float)):
            return (w_d * diffuse + w_s * specular,) * 3
        d = tuple(diffuse)
        s = tuple(specular)
        if len(d) != len(s):
            raise ValueError("diffuse 与 specular 分量数不一致")
        return tuple(w_d * di + w_s * si for di, si in zip(d, s))
