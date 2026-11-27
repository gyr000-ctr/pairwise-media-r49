import math

import pytest

import depthpeel

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _d():
    return depthpeel.DepthPeel(3)

CASES = [
    ("peel", lambda: _d().peel([0.5, 0.2, 0.9]) == [0.2, 0.5, 0.9]),
    ("strict", lambda: all(a < b for a, b in zip(_d().peel([0.3, 0.1, 0.2]), _d().peel([0.3, 0.1, 0.2])[1:]))),
    ("dedup", lambda: _d().peel([0.5, 0.5, 0.7]) == [0.5, 0.7]),
    ("cap", lambda: len(depthpeel.DepthPeel(2).peel([0.1, 0.2, 0.3])) == 2),
    ("empty", lambda: _d().peel([]) == []),
    ("count", lambda: (lambda d: (d.peel([0.1, 0.2]), d.count())[1] == 2)(_d())),
    ("bad_layers", lambda: _raises(lambda: depthpeel.DepthPeel(0))),
    ("bad_depth", lambda: _raises(lambda: _d().peel([-0.1]))),
    ("type", lambda: isinstance(_d().peel([0.1]), list)),
    ("det", lambda: _d().peel([0.4, 0.1]) == _d().peel([0.4, 0.1])),
    ("nan", lambda: _raises(lambda: _d().peel([float('nan')]))),
    ("no_mut", lambda: _d().peel([0.1, 0.2]) is not None),
    ("count_type", lambda: (_d().peel([0.1]), isinstance(_d().count(), int))[1]),
    ("one", lambda: _d().peel([0.7]) == [0.7]),
    ("no_sort_mut", lambda: (lambda xs: (_d().peel(xs), xs)[1] == [0.9, 0.1])([0.9, 0.1])),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
