import pytest

from pharmpy.basic import RandomNumberGenerator, Seed


def test_random_seed():
    seed = Seed()
    assert isinstance(seed.value, int)
    seed2 = Seed(None)
    assert isinstance(seed2.value, int)
    assert seed.value != seed2.value


@pytest.mark.parametrize(
    "obj,value",
    [
        (1, 1),
        (23.0, 23),
        (Seed(195), 195),
    ],
)
def test_good_seeds(obj, value):
    seed = Seed(obj)
    assert seed.value == value
    assert int(seed) == value


@pytest.mark.parametrize(
    "obj",
    [
        18.5,
        "myseed",
    ],
)
def test_bad_seeds(obj):
    with pytest.raises(ValueError):
        Seed(obj)


def test_seed_repr():
    seed = Seed(23)
    assert repr(seed) == "Seed(23)"


@pytest.mark.parametrize(
    "seed",
    [
        18912,
        Seed(19912),
        RandomNumberGenerator(987),
        [1, 2, 3],
    ],
)
def test_rng(seed):
    rng = RandomNumberGenerator(seed)
    rng.to_numpy()
