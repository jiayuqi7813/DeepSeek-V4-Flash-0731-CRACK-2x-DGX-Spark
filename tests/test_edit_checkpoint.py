from __future__ import annotations

import pytest

from dspark_crack.edit_checkpoint import parse_layers


def test_parse_layers() -> None:
    assert parse_layers("0,2-4,4", 8) == [0, 2, 3, 4]


def test_parse_layers_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        parse_layers("42-10", 43)
    with pytest.raises(ValueError):
        parse_layers("43", 43)
