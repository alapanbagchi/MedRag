"""Unit tests for scripts/medpat_fullvec_migrate.py (DDL + type detection).

No DB: asserts the migration targets full-precision vector(768) with
vector_ip_ops and is idempotent-safe on type detection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "medpat_fullvec_migrate.py"
_spec = importlib.util.spec_from_file_location("medpat_fullvec_migrate", _SCRIPT)
medpat_fullvec_migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(medpat_fullvec_migrate)


def test_ddl_widens_to_full_precision_vector():
    assert "vector(768)" in medpat_fullvec_migrate.ALTER_COLUMN
    assert "USING embedding::vector" in medpat_fullvec_migrate.ALTER_COLUMN
    assert "vector_ip_ops" in medpat_fullvec_migrate.CREATE_INDEX
    # the target schema must not keep the fp16 type/opclass
    assert "halfvec" not in medpat_fullvec_migrate.ALTER_COLUMN
    assert "halfvec" not in medpat_fullvec_migrate.CREATE_INDEX
    assert medpat_fullvec_migrate.DROP_INDEX.startswith("DROP INDEX IF EXISTS")


def test_is_vector_type():
    assert medpat_fullvec_migrate._is_vector_type("vector(768)")
    assert medpat_fullvec_migrate._is_vector_type("vector (768)")  # spacing
    assert not medpat_fullvec_migrate._is_vector_type("halfvec(768)")
    assert not medpat_fullvec_migrate._is_vector_type("")


def test_truncate_targets_chunk_embeddings():
    assert medpat_fullvec_migrate.TRUNCATE == "TRUNCATE medpat.chunk_embeddings"
