"""光照网格扩散：4 邻接（5 点 Laplace）迭代到收敛。

求解离散 Dirichlet 问题::

    4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = 0   （自由格）

边界格与 fixed 指定的固定格保持不变。默认使用几何多重网格 V-cycle
加速（V(1,1)，红-黑 Gauss-Seidel 平滑），对极小网格回退为直接
红-黑 Gauss-Seidel；两者都是"两次迭代差 <= tol 即收敛、iters 为
迭代上限"的迭代法。仅使用 Python 标准库。
"""

import math

__all__ = ["LightGrid", "LightGridError", "InvalidGridError",
           "InvalidFixedError", "InvalidConfigError"]


class LightGridError(ValueError):
    """lightgrid 所有参数/数据校验错误的基类（本身是 ValueError）。"""


class InvalidConfigError(LightGridError):
    """构造参数（迭代上限 / 容差）非法。"""


class InvalidGridError(LightGridError):
    """网格数据非法（尺寸、矩形性、元素类型、NaN/Inf）。"""


class InvalidFixedError(LightGridError):
    """固定格列表非法（坐标越界、长度不符、值非有限数）。"""


class _Level:
    __slots__ = ("n", "m", "red", "black", "interior")

    def __init__(self):
        self.n = 0
        self.m = 0
        self.red = ()
        self.black = ()
        self.interior = ()


def _color_lists(n):
    """生成 n×n 内部格的红/黑（棋盘奇偶）与全内部索引。"""
    red, black, interior = [], [], []
    for i in range(1, n - 1):
        base = i * n
        for j in range(1, n - 1):
            p = base + j
            interior.append(p)
            (red if (i + j) & 1 else black).append(p)
    return red, black, interior


def _build_level(n, m):
    """构造 n -> m（m ≈ (n+1)/2，粗网格点 2k 对齐细网格点）的一层。"""
    red, black, interior = _color_lists(n)
    lv = _Level()
    lv.n = n
    lv.m = m
    lv.red = red
    lv.black = black
    lv.interior = interior
    return lv


def _build_coarsest(n):
    lv = _Level()
    lv.n = n
    lv.m = 0
    red, black, interior = _color_lists(n)
    lv.red = red
    lv.black = black
    lv.interior = interior
    return lv


def _build_hierarchy(n, coarse_min):
    sizes = [n]
    while sizes[-1] > coarse_min:
        s = sizes[-1]
        # 保持粗网格为奇数尺寸，使粗边界点 2k 精确落在细边界上
        sizes.append(s // 2 + 1 if s % 2 == 0 else (s + 1) // 2)
    levels = [_build_level(sizes[k], sizes[k + 1]) for k in range(len(sizes) - 1)]
    levels.append(_build_coarsest(sizes[-1]))
    return levels


def _smooth(u, rhs, n, free_red, free_black, fixed_anchor, sweeps, track=False):
    """红-黑 Gauss-Seidel；自由格索引已剔除固定格，末尾重新锚定固定格。

    track=True 时额外返回最后一次扫描中自由格更新量的最大绝对值。
    """
    q = 0.25
    dmax = 0.0
    for s in range(sweeps):
        last = s == sweeps - 1
        for half in (free_red, free_black):
            for p in half:
                nv = (rhs[p] + u[p - n] + u[p + n] + u[p - 1] + u[p + 1]) * q
                if track and last:
                    d = nv - u[p]
                    if d < 0.0:
                        d = -d
                    if d > dmax:
                        dmax = d
                u[p] = nv
        for p, v in fixed_anchor:
            u[p] = v
    return dmax if track else None


def _residual(u, rhs, interior_free, n):
    r = [0.0] * (n * n)
    for p in interior_free:
        r[p] = rhs[p] - (4.0 * u[p] - u[p - n] - u[p + n] - u[p - 1] - u[p + 1])
    return r


def _restrict(rf, lv, fine_fixed):
    """限制残差并生成粗网格固定掩码；返回 (粗 rhs, 粗掩码, 粗固定锚点)。

    粗点 (ic,jc) 对应细点 (2ic,2jc)，采用 9 点全权重 stencil：
    中心 1/4、边邻 1/8、角邻 1/16。
    """
    m, n = lv.m, lv.n
    rc = [0.0] * (m * m)
    cm = bytearray(m * m)
    for ic in range(1, m - 1):
        fi = 2 * ic
        base = ic * m
        fbase = fi * n
        up = fbase - n
        dn = fbase + n
        for jc in range(1, m - 1):
            fj = 2 * jc
            acc = (
                rf[up + fj - 1] + 2.0 * rf[up + fj] + rf[up + fj + 1]
                + 2.0 * rf[fbase + fj - 1] + 4.0 * rf[fbase + fj]
                + 2.0 * rf[fbase + fj + 1]
                + rf[dn + fj - 1] + 2.0 * rf[dn + fj] + rf[dn + fj + 1]
            ) * 0.0625
            kc = base + jc
            rc[kc] = acc
            # 该粗点限制 stencil 的 9 个细格中只要有固定格，就把粗点的
            # 修正量钉为 0：保证固定格在每个层级都满足零修正边界条件。
            if (fine_fixed[up + fj - 1] or fine_fixed[up + fj]
                    or fine_fixed[up + fj + 1]
                    or fine_fixed[fbase + fj - 1] or fine_fixed[fbase + fj]
                    or fine_fixed[fbase + fj + 1]
                    or fine_fixed[dn + fj - 1] or fine_fixed[dn + fj]
                    or fine_fixed[dn + fj + 1]):
                cm[kc] = 1
    # 粗边界：修正量必须为 0
    for j in range(m):
        cm[j] = 1
        cm[(m - 1) * m + j] = 1
    for i in range(m):
        cm[i * m] = 1
        cm[i * m + m - 1] = 1
    anchor = [(k, 0.0) for k in range(m * m) if cm[k]]
    return rc, cm, anchor


def _prolong(ec, lv):
    """双线性延拓：细点 (i,j) 由粗点 (i//2, j//2) 邻域按奇偶插值。"""
    n, m = lv.n, lv.m
    ef = [0.0] * (n * n)
    for i in range(1, n - 1):
        ic, iodd = divmod(i, 2)
        base = i * n
        crow = ic * m
        nrow = (ic + 1) * m
        if iodd:
            for j in range(1, n - 1):
                jc, jodd = divmod(j, 2)
                if jodd:
                    ef[base + j] = 0.25 * (
                        ec[crow + jc] + ec[crow + jc + 1]
                        + ec[nrow + jc] + ec[nrow + jc + 1])
                else:
                    ef[base + j] = 0.5 * (ec[crow + jc] + ec[nrow + jc])
        else:
            for j in range(1, n - 1):
                jc, jodd = divmod(j, 2)
                if jodd:
                    ef[base + j] = 0.5 * (ec[crow + jc] + ec[crow + jc + 1])
                else:
                    ef[base + j] = ec[crow + jc]
    return ef


def _vcycle(u, rhs, levels, level_data, depth, nu1=1, nu2=1):
    """单层 V-cycle。

    level_data[k] = (限制传播掩码, 锚点, 自由红, 自由黑)。
    返回顶层（depth 层）后置平滑的最大更新量，用作两次迭代差的收敛判据。
    """
    lv = levels[depth]
    n = lv.n
    rmask, anchor, free_red, free_black = level_data[depth]
    _smooth(u, rhs, n, free_red, free_black, anchor, nu1)

    if depth == len(levels) - 1:
        return _smooth(u, rhs, n, free_red, free_black, anchor, 60, track=True)

    rf = _residual(u, rhs, free_red + free_black, n)
    rc, _, _ = _restrict(rf, lv, rmask)
    # 粗网格间距翻倍 => Galerkin 粗算子相对细 stencil 为 4 倍，rhs 层级标度 ×4
    for k in range(len(rc)):
        rc[k] *= 4.0

    ec = [0.0] * (lv.m * lv.m)
    _vcycle(ec, rc, levels, level_data, depth + 1, nu1, nu2)

    ef = _prolong(ec, lv)
    for p in free_red + free_black:
        u[p] += ef[p]
    for p, v in anchor:
        u[p] = v
    return _smooth(u, rhs, n, free_red, free_black, anchor, nu2, track=True)


class LightGrid:
    """4 邻接光照扩散求解器。

    Parameters
    ----------
    iters:
        迭代次数上限（正整数）。多重网格路径下每次 V-cycle 计为一次迭代。
    tol:
        收敛容差（正数）；相邻两次迭代全场最大绝对差 <= tol 即停止。
    """

    #: 网格边长达到该值才启用多重网格，否则直接红-黑 Gauss-Seidel。
    MG_MIN_SIZE = 17
    #: 多重网格最粗层边长上限。
    COARSE_MIN = 9

    #: 最近一次 solve 实际使用的迭代数（类级共享，跨实例可读）。
    _last_iterations = 0

    def __init__(self, iters, tol):
        if isinstance(iters, bool) or not isinstance(iters, int) or iters <= 0:
            raise InvalidConfigError(
                f"iters 必须为正整数，收到 {iters!r}")
        if isinstance(tol, bool) or not isinstance(tol, (int, float)):
            raise InvalidConfigError(f"tol 必须为正数，收到 {tol!r}")
        t = float(tol)
        if not math.isfinite(t) or t <= 0.0:
            raise InvalidConfigError(f"tol 必须为有限正数，收到 {tol!r}")
        self._iters_cap = iters
        self._tol = t
        # None 表示本实例尚未求解；0 是合法结果（无自由格时一次也不迭代）
        self._used_iters = None

    # ------------------------------------------------------------------
    def iterations(self):
        """实际迭代数。

        优先返回本实例上一次 solve 的计数（含 0，即无自由格的情形）；
        若本实例尚未求解，则回退到类级共享的"最近一次 solve"计数（默认 0）。
        返回 int。
        """
        if self._used_iters is not None:
            return self._used_iters
        return type(self)._last_iterations

    def solve(self, grid, fixed):
        """对给定初值网格与固定格求解到收敛。

        Parameters
        ----------
        grid:
            矩形二维 list，元素为 int/float 且全部有限；不会被修改。
        fixed:
            ``(row, col, value)`` 三元组序列，坐标必须在网格内，
            value 为有限数。固定格在迭代中保持该值。

        Returns
        -------
        list[list[float]]
            与 grid 同形的新二维列表。
        """
        rows, cols, flat = self._validate_grid(grid)
        explicit_anchor, explicit_mask = self._validate_fixed(fixed, rows, cols)

        # 边界格隐式固定：钉在输入值上，不参与更新（题面要求）。
        fixed_mask = bytearray(rows * cols)
        for j in range(cols):
            fixed_mask[j] = 1
            fixed_mask[(rows - 1) * cols + j] = 1
        for i in range(rows):
            fixed_mask[i * cols] = 1
            fixed_mask[i * cols + cols - 1] = 1

        # 仅供多重网格限制使用的"内部固定格"掩码：边界由各粗层边界单独处理，
        # 这里只保留显式指定的内部固定格，避免边界被过度传播到粗内部层。
        interior_fixed = bytearray(rows * cols)
        for p, v in explicit_anchor:
            r, c = divmod(p, cols)
            if 0 < r < rows - 1 and 0 < c < cols - 1:
                interior_fixed[p] = 1

        # 显式 fixed 叠加（fixed 内部的重复指定已在 _validate_fixed 拦截）
        fixed_anchor = []
        for p, v in explicit_anchor:
            fixed_mask[p] = 1
            fixed_anchor.append((p, v))
        # 其余固定格（未显式给值的边界）锚定到输入初值
        for p in range(rows * cols):
            if fixed_mask[p] and not explicit_mask[p]:
                fixed_anchor.append((p, flat[p]))
        # 以显式 fixed 值覆盖初值
        for p, v in explicit_anchor:
            flat[p] = v

        stride = cols
        rhs = [0.0] * (rows * cols)
        free_red, free_black = self._free_lists(rows, cols, fixed_mask)

        if not free_red and not free_black:
            # 没有任何自由格：无需迭代
            self._used_iters = 0
            return self._reshape(flat, rows, cols)

        if rows < self.MG_MIN_SIZE or cols < self.MG_MIN_SIZE or rows != cols:
            used = self._solve_gs(flat, rhs, stride, free_red, free_black, fixed_anchor)
        else:
            used = self._solve_mg(flat, rhs, rows, interior_fixed,
                                  fixed_anchor, free_red, free_black)
        self._used_iters = used
        type(self)._last_iterations = used
        return self._reshape(flat, rows, cols)

    # ------------------------------------------------------------------
    def _solve_gs(self, u, rhs, n, free_red, free_black, anchor):
        q = 0.25
        tol = self._tol
        cap = self._iters_cap
        used = 0
        for used in range(1, cap + 1):
            dmax = 0.0
            for half in (free_red, free_black):
                for p in half:
                    nv = (rhs[p] + u[p - n] + u[p + n] + u[p - 1] + u[p + 1]) * q
                    d = nv - u[p]
                    if abs(d) > dmax:
                        dmax = abs(d)
                    u[p] = nv
            for p, v in anchor:
                u[p] = v
            if dmax <= tol:
                break
        return used

    def _solve_mg(self, u, rhs, n, interior_fixed, anchor,
                  free_red, free_black):
        levels = _build_hierarchy(n, self.COARSE_MIN)

        # 每层准备 (限制传播掩码, 锚点, 自由红, 自由黑)。
        # - rmask：仅含内部固定格，供 _restrict 向粗层传播（边界由粗边界单独钉 0）；
        # - free_*：基于完整固定掩码（边界 + 内部固定格），决定平滑更新哪些格。
        level_data = [(interior_fixed, anchor, free_red, free_black)]
        rmask = interior_fixed
        for k, lv in enumerate(levels[:-1]):
            _, cm, canchor = _restrict([0.0] * (lv.n * lv.n), lv, rmask)
            rmask = cm  # 下一层的传播掩码（已含其粗边界）
            coarse = levels[k + 1]  # 边长 lv.m，与 cm 同尺寸
            fr = [p for p in coarse.red if not cm[p]]
            fb = [p for p in coarse.black if not cm[p]]
            level_data.append((cm, canchor, fr, fb))

        tol = self._tol
        cap = self._iters_cap
        used = 0
        for used in range(1, cap + 1):
            # _vcycle 顶层后置平滑的最大更新量即本次迭代的全场最大两次差
            dmax = _vcycle(u, rhs, levels, level_data, 0)
            if dmax <= tol:
                break
        return used

    # ------------------------------------------------------------------
    @staticmethod
    def _free_lists(rows, cols, fixed_mask):
        free_red, free_black = [], []
        for i in range(rows):
            base = i * cols
            for j in range(cols):
                p = base + j
                if fixed_mask[p]:
                    continue
                if (i + j) & 1:
                    free_red.append(p)
                else:
                    free_black.append(p)
        return free_red, free_black

    @staticmethod
    def _reshape(flat, rows, cols):
        return [flat[i * cols:(i + 1) * cols] for i in range(rows)]

    @staticmethod
    def _validate_grid(grid):
        if not isinstance(grid, list) or len(grid) == 0:
            raise InvalidGridError("grid 必须为非空二维 list")
        rows = len(grid)
        cols = None
        flat = []
        for r, row in enumerate(grid):
            if not isinstance(row, list):
                raise InvalidGridError(f"grid 第 {r} 行不是 list")
            if cols is None:
                cols = len(row)
                if cols == 0:
                    raise InvalidGridError("grid 不允许空行")
            elif len(row) != cols:
                raise InvalidGridError(
                    f"grid 不是矩形：第 0 行宽 {cols}，第 {r} 行宽 {len(row)}")
            for c, v in enumerate(row):
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise InvalidGridError(
                        f"grid[{r}][{c}] 不是数值类型：{v!r}")
                fv = float(v)
                if not math.isfinite(fv):
                    raise InvalidGridError(
                        f"grid[{r}][{c}] 不是有限数：{v!r}")
                flat.append(fv)
        return rows, cols, flat

    @staticmethod
    def _validate_fixed(fixed, rows, cols):
        if not isinstance(fixed, (list, tuple)):
            raise InvalidFixedError("fixed 必须为 (row, col, value) 序列")
        mask = bytearray(rows * cols)
        anchor = []
        for idx, item in enumerate(fixed):
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise InvalidFixedError(
                    f"fixed[{idx}] 必须是长度为 3 的 (row, col, value)")
            r, c, v = item
            if isinstance(r, bool) or not isinstance(r, int) or \
               isinstance(c, bool) or not isinstance(c, int):
                raise InvalidFixedError(
                    f"fixed[{idx}] 坐标必须为整数：{(r, c)!r}")
            if not (0 <= r < rows and 0 <= c < cols):
                raise InvalidFixedError(
                    f"fixed[{idx}] 坐标 ({r}, {c}) 越界，网格尺寸 {rows}x{cols}")
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise InvalidFixedError(
                    f"fixed[{idx}] 值不是数值类型：{v!r}")
            fv = float(v)
            if not math.isfinite(fv):
                raise InvalidFixedError(
                    f"fixed[{idx}] 值不是有限数：{v!r}")
            p = r * cols + c
            if mask[p]:
                raise InvalidFixedError(
                    f"fixed[{idx}] 重复指定格 ({r}, {c})")
            mask[p] = 1
            anchor.append((p, fv))
        return anchor, mask
