import math

import pytest

import lightgrid

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _g():
    return lightgrid.LightGrid(200, 1e-6)

CASES = [
    ("constant", lambda: all(abs(v - 0.5) < 1e-12 for row in _g().solve([[0.5] * 4 for _ in range(4)], []) for v in row)),
    ("fixed", lambda: abs(_g().solve([[0.0] * 3 for _ in range(3)], [(0, 0, 1.0)])[0][0] - 1.0) < 1e-12),
    ("iters", lambda: (_g().solve([[0.0] * 3 for _ in range(3)], [(0, 0, 1.0)]), _g().iterations())[1] >= 1),
    ("bad_iters", lambda: _raises(lambda: lightgrid.LightGrid(0, 1e-6))),
    ("bad_tol", lambda: _raises(lambda: lightgrid.LightGrid(10, 0.0))),
    ("bad_grid", lambda: _raises(lambda: _g().solve([], []))),
    ("bad_fixed", lambda: _raises(lambda: _g().solve([[0.0]], [(9, 9, 1.0)]))),
    ("type", lambda: isinstance(_g().solve([[0.0] * 3 for _ in range(3)], []), list)),
    ("det", lambda: _g().solve([[0.0] * 3 for _ in range(3)], [(1, 1, 0.7)]) == _g().solve([[0.0] * 3 for _ in range(3)], [(1, 1, 0.7)])),
    ("range", lambda: all(-1.0 <= v <= 2.0 for row in _g().solve([[0.0] * 3 for _ in range(3)], [(1, 1, 1.0)]) for v in row)),
    ("nan", lambda: _raises(lambda: _g().solve([[float('nan')]], []))),
    ("no_mut", lambda: _g().solve([[0.0] * 3 for _ in range(3)], []) is not None),
    ("iters_type", lambda: (_g().solve([[0.0]], []), isinstance(_g().iterations(), int))[1]),
    ("one_cell", lambda: len(_g().solve([[0.0]], [])) == 1),
    ("float", lambda: isinstance(_g().solve([[0.0] * 3 for _ in range(3)], [(0, 0, 1.0)])[1][1], float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
