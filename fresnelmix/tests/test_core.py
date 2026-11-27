import math

import pytest

import fresnelmix

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _f():
    return fresnelmix.FresnelMix(1.5)

CASES = [
    ("f0", lambda: abs(_f().f0() - 0.04) < 1e-12),
    ("normal", lambda: abs(_f().reflectance(1.0) - 0.04) < 1e-12),
    ("grazing", lambda: _f().reflectance(0.0) > 0.99),
    ("monotone", lambda: _f().reflectance(0.5) <= _f().reflectance(0.2)),
    ("range", lambda: 0.0 <= _f().reflectance(0.7) <= 1.0),
    ("mix_pair", lambda: all(abs(a - b) < 1e-12 for a, b in zip(_f().mix(1.0, 0.0, 1.0), (0.96, 0.96, 0.96)))),
    ("bad_ior", lambda: _raises(lambda: fresnelmix.FresnelMix(1.0))),
    ("bad_cos", lambda: _raises(lambda: _f().reflectance(1.5))),
    ("bad_cos_neg", lambda: _raises(lambda: _f().reflectance(-0.1))),
    ("type", lambda: isinstance(_f().reflectance(0.5), float)),
    ("det", lambda: _f().reflectance(0.5) == _f().reflectance(0.5)),
    ("nan", lambda: _raises(lambda: _f().reflectance(float('nan')))),
    ("no_mut", lambda: _f().mix((0.2, 0.2, 0.2), (0.8, 0.8, 0.8), 0.5) is not None),
    ("mix_type", lambda: isinstance(_f().mix((0.2, 0.2, 0.2), (0.8, 0.8, 0.8), 0.5), tuple)),
    ("f0_type", lambda: isinstance(_f().f0(), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
