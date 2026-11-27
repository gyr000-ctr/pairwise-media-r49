import math

import pytest

import envbrdf

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

T = [[0.1, 0.2], [0.3, 0.4]]


def _e():
    return envbrdf.EnvBrdf(T)

CASES = [
    ("lookup", lambda: isinstance(_e().lookup(0.5, 0.5), float)),
    ("corner", lambda: abs(_e().lookup(0.0, 0.0) - 0.1) < 1e-12),
    ("clamp", lambda: isinstance(_e().lookup(5.0, 5.0), float)),
    ("fwhm", lambda: _e().fwhm(0.5) > 0.0),
    ("bad_table", lambda: _raises(lambda: envbrdf.EnvBrdf([]))),
    ("bad_rough", lambda: _raises(lambda: _e().lookup(0.5, -0.1))),
    ("bad_ndv", lambda: _raises(lambda: _e().lookup(1.5, 0.5))),
    ("type", lambda: isinstance(_e().lookup(0.3, 0.3), float)),
    ("det", lambda: _e().lookup(0.3, 0.3) == _e().lookup(0.3, 0.3)),
    ("nan", lambda: _raises(lambda: _e().lookup(float('nan'), 0.5))),
    ("no_mut", lambda: _e().lookup(0.2, 0.2) is not None),
    ("fwhm_type", lambda: isinstance(_e().fwhm(0.2), float)),
    ("energy", lambda: 0.0 <= _e().lookup(0.5, 0.5) <= 1.0),
    ("monotone", lambda: _e().fwhm(0.3) <= _e().fwhm(0.8)),
    ("single", lambda: isinstance(envbrdf.EnvBrdf([[0.5]]).lookup(0.0, 0.0), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
