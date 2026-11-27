"""次表面散射偶极近似（dipole）径向 profile，仅使用 Python 标准库。

数学约定
--------
输入的 ``sigma_a`` / ``sigma_s`` 视为已约化（reduced）的吸收与散射系数。

偶极子采用 Jensen et al. 2001 的解析公式，并在 ``sigma_t' = 1`` 的无量纲
单位制下逐点计算（物理量加撇表示约化量）：

    sigma_t' = sigma_a + sigma_s
    alpha'   = sigma_s / sigma_t'                 反照率
    q        = sqrt(3 * sigma_a / sigma_t')       = sigma_tr / sigma_t'
    D        = 1 / 3                               无量纲扩散系数
    z_v      = 1                                   实光源深度
    z_r      = z_v + 2 A D                         虚光源高度

边界两侧折射率相同（构造器未给出 IOR，取 eta = 1），Fresnel 反射项

    F_dr = -1.440/eta^2 + 0.710/eta + 0.668 + 0.0636*eta
    A    = (1 + F_dr) / (1 - F_dr)

无量纲半径 ``x = r / sigma_t'``。所有系数同时乘以 c、距离同时乘以 c 时
x 不变，即平均自由程参数对 profile 作线性缩放时形状保持不变。

未归一化的偶极径向出射度：

    g(x) = alpha'/(4*pi) * sum_h  h * (q + 1/d) * exp(-q*d) / d^2
           d = sqrt(x^2 + h^2),  h 取 z_v 与 z_r

其精确径向积分（变量替换 d = sqrt(x^2+h^2)，即 x dx = d dd）：

    J_h = 2*pi*h * [ exp(-q*h) + E1(q*h) ]

其中 E1 为一阶指数积分（两项分别来自 q 与 1/d 部分：
``int_h^inf q e^-qd dd = e^-qh``、``int_h^inf e^-qd/d dd = E1(qh)``）。
于是

    I = alpha'/(4*pi) * (J_zv + J_zr)
      = alpha'/2 * [ z_v B(z_v) + z_r B(z_r) ],  B(h) = exp(-q*h) + E1(q*h)

归一化 profile P(x) = g(x) / I 满足

    integral_0^inf 2*pi*x * P(x) dx = 1
"""

import math

__all__ = ["SssProfile"]

_EULER_GAMMA = 0.5772156649015329


def _fresnel_dr(eta):
    """偶极边界处的漫反射 Fresnel 项（Jensen 2001 近似式）。"""
    return -1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta


def _e1(x):
    """一阶指数积分 E1(x) = integral_x^inf exp(-t)/t dt，x > 0。

    x <= 1 用幂级数，x > 1 用 Lentz 算法求连分式
    E1(x) = exp(-x) / (x + 1/(1 + 1/(x + 2/(1 + 2/(x + ...)))))。
    """
    if x <= 1.0:
        # E1(x) = -gamma - ln(x) - sum_{k=1}^inf (-x)^k / (k * k!)
        total = 0.0
        xk = 1.0  # xk 递推保存 x^k / k!，逐项构造 (-x)^k / (k * k!)
        k = 1
        while True:
            xk *= -x / k
            term = xk / k
            total += term
            if abs(term) <= 1e-17:
                break
            k += 1
        return -_EULER_GAMMA - math.log(x) - total

    # Lentz 连分式：b0 = x；a_{2k-1}=k, b_{2k-1}=1；a_{2k}=k, b_{2k}=x
    tiny = 1e-300
    f = x if x != 0.0 else tiny
    c = f
    d = 0.0
    k = 1
    while True:
        # 第 2k-1 层：a = k, b = 1
        d = 1.0 + k * d
        c = 1.0 + k / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta_j = c * d
        f *= delta_j
        # 第 2k 层：a = k, b = x
        d = x + k * d
        c = x + k / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta_k = c * d
        f *= delta_k
        if abs(delta_j * delta_k - 1.0) < 1e-16:
            break
        k += 1
        if k > 10000:  # pragma: no cover - 理论上不会触发
            raise RuntimeError("E1 continued fraction did not converge")
    return math.exp(-x) / f


def _check_coefficient(value, name):
    """散射系数必须是有限正实数；sigma_a / sigma_s 报错信息可区分。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("%s must be a real number, got %r" % (name, value))
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(
            "%s must be a finite positive number, got %r" % (name, value)
        )
    return result


# Gauss-Legendre 节点/权重缓存：n -> (nodes, weights)
_GL_CACHE = {}


def _gauss_legendre(n):
    """返回 n 点 Gauss-Legendre 求积的节点与权重（区间 (-1, 1)）。

    节点由 Legendre 多项式递推 + Newton 迭代求出，结果按 n 缓存。
    仅依赖标准库 math。
    """
    cached = _GL_CACHE.get(n)
    if cached is not None:
        return cached

    pi = math.pi
    cos = math.cos
    roots = [0.0] * n
    weights = [0.0] * n

    def legendre(value, order):
        p_prev = 1.0
        p_curr = value
        for k in range(2, order + 1):
            p_prev, p_curr = (
                p_curr,
                ((2.0 * k - 1.0) * value * p_curr - (k - 1.0) * p_prev) / k,
            )
        return p_prev, p_curr

    for i in range(n // 2):
        # P_n 第 i+1 个根的初值（靠近 +1 端）
        x = cos(pi * (i + 0.75) / (n + 0.5))
        for _ in range(100):
            p_prev, p_curr = legendre(x, n)
            # P_n'(x) = n * (x P_n(x) - P_{n-1}(x)) / (x^2 - 1)
            derivative = n * (x * p_curr - p_prev) / (x * x - 1.0)
            x_new = x - p_curr / derivative
            if abs(x_new - x) <= 1e-15:
                x = x_new
                break
            x = x_new

        p_prev, p_curr = legendre(x, n)
        derivative = n * (x * p_curr - p_prev) / (x * x - 1.0)
        weight = 2.0 / ((1.0 - x * x) * derivative * derivative)

        mirror = n - 1 - i
        roots[i] = -x
        roots[mirror] = x
        weights[i] = weight
        weights[mirror] = weight

    if n % 2:  # 奇数阶时中点 x = 0 也是根
        mid = n // 2
        roots[mid] = 0.0
        p_prev, _p_curr = legendre(0.0, n)  # p_prev = P_{n-1}(0)
        derivative = -n * p_prev            # P_n'(0) = -n P_{n-1}(0)
        weights[mid] = 2.0 / (derivative * derivative)

    cached = (tuple(roots), tuple(weights))
    _GL_CACHE[n] = cached
    return cached


class SssProfile:
    """次表面散射偶极近似径向 profile。

    Parameters
    ----------
    sigma_a:
        吸收系数（有限正实数），非法值抛出 ``ValueError``，信息中带
        ``sigma_a`` 字样。
    sigma_s:
        （约化）散射系数（有限正实数），非法值抛出 ``ValueError``，信息中
        带 ``sigma_s`` 字样，可与 sigma_a 的报错区分。
    """

    __slots__ = (
        "sigma_a",
        "sigma_s",
        "_sigma_t",
        "_alpha",
        "_q",
        "_zr",
        "_norm",
        "_total",
    )

    def __init__(self, sigma_a, sigma_s):
        sa = _check_coefficient(sigma_a, "sigma_a")
        ss = _check_coefficient(sigma_s, "sigma_s")

        st = sa + ss
        alpha = ss / st
        # sigma_tr / sigma_t' = sqrt(3 * sigma_a / sigma_t')
        q = math.sqrt(3.0 * sa / st)

        # eta = 1：构造器未提供折射率，边界两侧介质相同
        fdr = _fresnel_dr(1.0)
        boundary_a = (1.0 + fdr) / (1.0 - fdr)
        z_v = 1.0
        z_r = z_v + 2.0 * boundary_a / 3.0

        # 单源径向积分闭式 J_h = 2*pi*h * (exp(-q*h) + E1(q*h))
        b_v = math.exp(-q * z_v) + _e1(q * z_v)
        b_r = math.exp(-q * z_r) + _e1(q * z_r)
        # 未归一化 g 的总通量 I = alpha'/(4*pi) * (J_zv + J_zr)
        #                      = alpha'/2 * (z_v B_v + z_r B_r)
        total = 0.5 * alpha * (z_v * b_v + z_r * b_r)

        self.sigma_a = sa
        self.sigma_s = ss
        self._sigma_t = st
        self._alpha = alpha
        self._q = q
        self._zr = z_r
        self._total = total
        # 形状函数 g_shape（不含 alpha'/(4*pi)）的归一化乘子：
        # P = [alpha'/(4*pi)/I] * g_shape = 1/[2*pi*(z_v B_v + z_r B_r)] * g_shape
        self._norm = 1.0 / (2.0 * math.pi * (z_v * b_v + z_r * b_r))

    def profile(self, r):
        """返回径向距离 ``r`` 处的归一化 profile 值（float， >= 0）。

        r 为负数或 NaN 时抛出 ``ValueError``；r = +inf 返回 0.0。
        """
        if isinstance(r, bool) or not isinstance(r, (int, float)):
            raise TypeError("r must be a real number, got %r" % (r,))
        if r < 0.0 or r != r:  # 负数或 NaN
            raise ValueError("r must be non-negative and finite, got %r" % (r,))
        if r == float("inf"):
            return 0.0

        q = self._q
        zr = self._zr
        x = float(r) / self._sigma_t
        x2 = x * x

        d_r = math.sqrt(x2 + 1.0)          # sqrt(x^2 + z_v^2), z_v = 1
        d_v = math.sqrt(x2 + zr * zr)      # sqrt(x^2 + z_r^2)
        dr2 = d_r * d_r
        dv2 = d_v * d_v
        # 实光源项带因子 z_v = 1，虚光源项带因子 z_r
        term_r = math.exp(-q * d_r) * (q / dr2 + 1.0 / (dr2 * d_r))
        term_v = (
            zr
            * math.exp(-q * d_v)
            * (q / dv2 + 1.0 / (dv2 * d_v))
        )
        return self._norm * (term_r + term_v)

    def _source_integral(self, h, roots, weights, cut=80.0):
        """单偶极源的径向积分 2*pi*int_0^inf x*g_h(x) dx。

        变量替换 d = sqrt(x^2 + h^2)（x dx = d dd）后拆为两部分：

        * q 项 ``2*pi*h*int_h^inf q e^-qd dd`` 为纯指数，用 Gauss-Legendre
          数值求积（独立于解析归一化常数的实际数值计算路径）；
        * 1/d 项 ``2*pi*h*int_h^inf e^-qd/d dd = 2*pi*h*E1(qh)`` 含缓变
          尾部，直接用指数积分闭式，保证任意 q 下稳定到机器精度。

        d 从 h(>= 1) 起，避开 x = 0 处的形状奇点；积分截断到 h + cut/q。
        """
        q = self._q
        lo = h
        span = cut / q
        i_q = 0.0
        for node, weight in zip(roots, weights):
            d = lo + 0.5 * span * (node + 1.0)
            i_q += weight * q * math.exp(-q * d)
        i_q *= 0.5 * span
        return 2.0 * math.pi * h * (i_q + _e1(q * h))

    def normalize(self, samples):
        """归一化校验：数值积分 int_0^inf 2*pi*x*P(x) dx，应返回 1.0。

        在 d 域对两个偶极源分别求积（见 :meth:`_source_integral`），
        再除以构造时得到的解析总通量。返回 float。``samples`` 为
        Gauss-Legendre 节点数，须为 >= 2 的整数。
        """
        if isinstance(samples, bool) or not isinstance(samples, int):
            samples = int(samples)
        if samples < 2:
            raise ValueError("samples must be an integer >= 2")

        roots, weights = _gauss_legendre(samples)

        # 未归一化 g 的数值总通量：alpha'/(4*pi) * (J_zv + J_zr)
        j_v = self._source_integral(1.0, roots, weights)
        j_r = self._source_integral(self._zr, roots, weights)
        total_numeric = self._alpha / (4.0 * math.pi) * (j_v + j_r)

        # 与构造时的解析总通量之比；归一化正确时恒为 1.0
        return total_numeric / self._total
