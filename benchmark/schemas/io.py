"""Dataset I/O: JSON persistence, content-hash ids, parent lineage.

Every dataset artifact carries a `dataset_id` that is a content hash of its own
payload (excluding the id field itself), so identical content always produces
the identical id — reruns are idempotent and reports are traceable back to the
exact configs/traces/records that produced them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel

_ID_FIELDS = ("dataset_id", "board_id", "report_id")


def content_hash(model: BaseModel) -> str:
    """Deterministic 16-hex-char hash of the model's content, ignoring id fields."""
    payload = model.model_dump(mode="json")
    for field in _ID_FIELDS:
        payload.pop(field, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def stamp_dataset_id[M: BaseModel](model: M) -> M:
    """Return a copy with its id field set to the content hash."""
    for field in _ID_FIELDS:
        if field in type(model).model_fields:
            return model.model_copy(update={field: content_hash(model)})
    raise TypeError(f"{type(model).__name__} has no id field ({', '.join(_ID_FIELDS)})")


def derive[M: BaseModel](child: M, parent: BaseModel) -> M:
    """Stamp `child`'s id and link it to `parent` via parent_dataset_id."""
    if "parent_dataset_id" not in type(child).model_fields:
        raise TypeError(f"{type(child).__name__} has no parent_dataset_id field")
    parent_id = getattr(parent, "dataset_id", None)
    if not parent_id:
        raise ValueError("parent has no stamped dataset_id — stamp it first")
    child = child.model_copy(update={"parent_dataset_id": parent_id})
    return stamp_dataset_id(child)


def save(model: BaseModel, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n")
    return path


def load[M: BaseModel](cls: type[M], path: str | Path) -> M:
    return cls.model_validate_json(Path(path).read_text())
