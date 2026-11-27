import math

import pytest

import sssprofile

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _s():
    return sssprofile.SssProfile(0.1, 1.0)

CASES = [
    ("decreasing", lambda: _s().profile(0.5) <= _s().profile(0.1)),
    ("positive", lambda: _s().profile(0.1) > 0.0),
    ("normalized", lambda: abs(_s().normalize(256) - 1.0) < 1e-9),
    ("bad_sigma_a", lambda: _raises(lambda: sssprofile.SssProfile(0.0, 1.0))),
    ("bad_sigma_s", lambda: _raises(lambda: sssprofile.SssProfile(0.1, 0.0))),
    ("bad_r", lambda: _raises(lambda: _s().profile(-0.1))),
    ("type", lambda: isinstance(_s().profile(0.2), float)),
    ("det", lambda: _s().profile(0.2) == _s().profile(0.2)),
    ("nan", lambda: _raises(lambda: _s().profile(float('nan')))),
    ("no_mut", lambda: _s().profile(0.3) is not None),
    ("scale", lambda: abs(sssprofile.SssProfile(0.1, 1.0).profile(0.2) - sssprofile.SssProfile(0.2, 2.0).profile(0.4)) < 1e-9),
    ("normalize_type", lambda: isinstance(_s().normalize(64), float)),
    ("big_r", lambda: abs(_s().profile(50.0)) < 1e-6),
    ("zero_r", lambda: _s().profile(0.0) > 0.0),
    ("monotone2", lambda: _s().profile(1.0) <= _s().profile(0.5)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
