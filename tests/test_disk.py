import pytest
from jevrec.utils.disk import free_gb, require_free_gb, DiskSpaceError


def test_free_gb_is_positive(tmp_path):
    assert free_gb(str(tmp_path)) > 0


def test_require_free_gb_passes_when_enough(tmp_path):
    require_free_gb(0.001, str(tmp_path))


def test_require_free_gb_raises_when_short(tmp_path):
    with pytest.raises(DiskSpaceError, match="below the required"):
        require_free_gb(10**6, str(tmp_path))
