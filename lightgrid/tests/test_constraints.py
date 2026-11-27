"""题面硬约束测试：解析解对照、边界不变、常量不变、性能、可区分报错。"""

import math
import time

import pytest

import lightgrid
from lightgrid import (
    LightGrid,
    InvalidConfigError,
    InvalidFixedError,
    InvalidGridError,
)


def _harmonic(n):
    """u(x,y)=x^2-y^2 为调和函数，5 点 Laplace 离散可精确再现。"""
    h = 1.0 / (n - 1)
    grid = [[0.0] * n for _ in range(n)]
    fixed = []
    for i in range(n):
        for j in range(n):
            if i in (0, n - 1) or j in (0, n - 1):
                v = (i * h) ** 2 - (j * h) ** 2
                grid[i][j] = v
                fixed.append((i, j, v))
    return grid, fixed, h


def _analytic_error(result, n, h):
    e = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            exact = (i * h) ** 2 - (j * h) ** 2
            e = max(e, abs(result[i][j] - exact))
    return e


@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_analytic_laplace_agreement(n):
    grid, fixed, h = _harmonic(n)
    result = LightGrid(500, 1e-9).solve(grid, fixed)
    assert _analytic_error(result, n, h) <= 1e-6


def test_boundary_cells_unchanged_cellwise():
    n = 48
    grid, fixed, _ = _harmonic(n)
    result = LightGrid(500, 1e-10).solve(grid, fixed)
    drift = 0.0
    for i in range(n):
        for j in range(n):
            if i in (0, n - 1) or j in (0, n - 1):
                drift = max(drift, abs(result[i][j] - grid[i][j]))
    assert drift <= 1e-12


def test_interior_fixed_cell_unchanged():
    grid = [[0.0] * 10 for _ in range(10)]
    result = LightGrid(500, 1e-12).solve(grid, [(5, 5, 0.77)])
    assert abs(result[5][5] - 0.77) <= 1e-12


@pytest.mark.parametrize("size", [4, 16, 33])
def test_constant_initial_value_invariant(size):
    grid = [[0.5] * size for _ in range(size)]
    result = LightGrid(500, 1e-12).solve(grid, [])
    assert all(abs(v - 0.5) <= 1e-12 for row in result for v in row)


def test_convergence_criterion_two_iteration_diff():
    # 容差很松时应在很少迭代内停止；很紧时迭代更多（受上限约束）
    grid, fixed, _ = _harmonic(32)
    loose = LightGrid(500, 1e-3)
    tight = LightGrid(500, 1e-10)
    loose.solve(grid, fixed)
    tight.solve(grid, fixed)
    assert loose.iterations() <= tight.iterations()
    assert tight.iterations() >= 1


def test_iteration_cap_is_hard_limit():
    grid, fixed, _ = _harmonic(64)
    cap = 3
    g = LightGrid(cap, 1e-15)  # 几乎不可能在 3 次内达到
    g.solve(grid, fixed)
    assert g.iterations() == cap


def test_256_performance_under_700ms():
    grid, fixed, h = _harmonic(256)
    g = LightGrid(500, 1e-7)
    t0 = time.perf_counter()
    result = g.solve(grid, fixed)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms <= 700.0
    # 性能测试同时保证结果仍正确
    assert _analytic_error(result, 256, h) <= 1e-6


@pytest.mark.parametrize("exc, fn", [
    (InvalidConfigError, lambda: LightGrid(0, 1e-6)),
    (InvalidConfigError, lambda: LightGrid(-1, 1e-6)),
    (InvalidConfigError, lambda: LightGrid(1.5, 1e-6)),
    (InvalidConfigError, lambda: LightGrid(10, 0.0)),
    (InvalidConfigError, lambda: LightGrid(10, -1e-6)),
    (InvalidConfigError, lambda: LightGrid(10, float("nan"))),
    (InvalidGridError, lambda: LightGrid(10, 1e-6).solve([], [])),
    (InvalidGridError, lambda: LightGrid(10, 1e-6).solve([[1.0, 2.0], [3.0]], [])),
    (InvalidGridError, lambda: LightGrid(10, 1e-6).solve(
        [[1.0, "x"], [0.0, 0.0]], [])),
    (InvalidGridError, lambda: LightGrid(10, 1e-6).solve(
        [[float("nan")]], [])),
    (InvalidGridError, lambda: LightGrid(10, 1e-6).solve(
        [[float("inf")]], [])),
    (InvalidFixedError, lambda: LightGrid(10, 1e-6).solve(
        [[0.0]], [(9, 9, 1.0)])),
    (InvalidFixedError, lambda: LightGrid(10, 1e-6).solve(
        [[0.0] * 3 for _ in range(3)], [(1.0, 1, 1.0)])),
    (InvalidFixedError, lambda: LightGrid(10, 1e-6).solve(
        [[0.0] * 3 for _ in range(3)], [(1, 1, float("inf"))])),
    (InvalidFixedError, lambda: LightGrid(10, 1e-6).solve(
        [[0.0] * 3 for _ in range(3)], [(1, 1, 0.2), (1, 1, 0.3)])),
    (InvalidFixedError, lambda: LightGrid(10, 1e-6).solve(
        [[0.0] * 3 for _ in range(3)], [(1, 1)])),
])
def test_distinguishable_errors(exc, fn):
    with pytest.raises(exc):
        fn()


def test_error_types_are_value_errors():
    for exc in (InvalidConfigError, InvalidGridError, InvalidFixedError):
        assert issubclass(exc, ValueError)


def test_input_grid_not_mutated():
    import copy
    grid = [[0.0] * 5 for _ in range(5)]
    snapshot = copy.deepcopy(grid)
    LightGrid(300, 1e-10).solve(grid, [(0, 0, 1.0)])
    assert grid == snapshot


def test_deterministic():
    grid, fixed, _ = _harmonic(24)
    a = LightGrid(300, 1e-10).solve(grid, fixed)
    b = LightGrid(300, 1e-10).solve(grid, fixed)
    assert a == b


def test_nonsquare_grid_uses_gs_and_converges():
    rows, cols = 4, 6
    result = LightGrid(500, 1e-10).solve([[0.0] * cols for _ in range(rows)], [])
    residual = 0.0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            residual = max(
                residual,
                abs(4 * result[i][j] - result[i - 1][j] - result[i + 1][j]
                    - result[i][j - 1] - result[i][j + 1]))
    assert residual <= 1e-8


def test_single_cell_grid():
    # 1x1 全为边界（隐式固定），原样返回且不迭代
    g = LightGrid(10, 1e-6)
    result = g.solve([[0.375]], [])
    assert result == [[0.375]]
    assert g.iterations() == 0


def _free_residual(result, n, fixed_set):
    m = 0.0
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if (i, j) in fixed_set:
                continue
            m = max(
                m,
                abs(4 * result[i][j] - result[i - 1][j] - result[i + 1][j]
                    - result[i][j - 1] - result[i][j + 1]))
    return m


@pytest.mark.parametrize("fixed", [
    [(20, 30, 0.9), (40, 15, -0.5), (50, 50, 0.33), (10, 10, 0.7)],   # even-even
    [(21, 31, 0.9), (41, 15, -0.5), (51, 51, 0.33), (11, 11, 0.7)],   # odd-odd
    [(20, 31, 0.9), (41, 15, -0.5), (50, 51, 0.33), (11, 10, 0.7)],   # mixed parity
])
def test_mg_converges_with_interior_fixed_cells(fixed):
    # 65x65 走多重网格路径；多内部固定格（含发散回归）必须收敛且固定格保持
    n = 65
    grid = [[0.0] * n for _ in range(n)]
    grid[0] = [1.0] * n
    g = LightGrid(2000, 1e-10)
    result = g.solve(grid, fixed)
    fixed_set = {(r, c) for r, c, _ in fixed}
    assert g.iterations() < 2000  # 真正收敛，而不是撞上迭代上限
    assert _free_residual(result, n, fixed_set) <= 1e-8
    for r, c, v in fixed:
        assert abs(result[r][c] - v) <= 1e-12


def test_mg_matches_gauss_seidel_truth():
    # 多重网格解应与充分迭代的 Gauss-Seidel 解一致
    n = 33
    grid = [[0.0] * n for _ in range(n)]
    grid[0] = [1.0] * n
    fixed = [(11, 7, 0.9), (21, 15, -0.5), (25, 25, 0.33), (5, 5, 0.7)]

    # 参考：直接红-黑 Gauss-Seidel 到极紧容差
    u = [float(v) for row in grid for v in row]
    mask = bytearray(n * n)
    for j in range(n):
        mask[j] = 1
        mask[(n - 1) * n + j] = 1
    for i in range(n):
        mask[i * n] = 1
        mask[i * n + n - 1] = 1
    anchor = []
    for r, c, v in fixed:
        p = r * n + c
        mask[p] = 1
        u[p] = v
        anchor.append((p, v))
    for p in range(n * n):
        if mask[p] and p not in {q for q, _ in anchor}:
            anchor.append((p, u[p]))
    free = [p for p in range(n * n) if not mask[p]]
    red = [p for p in free if ((p // n) + (p % n)) & 1]
    black = [p for p in free if p not in red]
    for _ in range(20000):
        d = 0.0
        for half in (red, black):
            for p in half:
                nv = (u[p - n] + u[p + n] + u[p - 1] + u[p + 1]) * 0.25
                d = max(d, abs(nv - u[p]))
                u[p] = nv
        for p, v in anchor:
            u[p] = v
        if d <= 1e-12:
            break

    result = LightGrid(2000, 1e-12).solve(grid, fixed)
    flat = [v for row in result for v in row]
    assert max(abs(a - b) for a, b in zip(flat, u)) <= 1e-8
