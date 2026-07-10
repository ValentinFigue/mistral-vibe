from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
import pytest

from vibe.core.graph.cache import CacheStore
from vibe.core.graph.operators import operator


class IntVal(BaseModel):
    n: int


@operator(name="tst_const")
async def const(value: int) -> IntVal:
    return IntVal(n=value)


@operator(name="tst_add")
async def add(a: IntVal, b: IntVal) -> IntVal:
    return IntVal(n=a.n + b.n)


@pytest.fixture
def cache(tmp_path: Path) -> CacheStore:
    store = CacheStore(tmp_path / "cache.sqlite")
    yield store
    store.close()
