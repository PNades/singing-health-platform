"""Object storage for the collection site (spec 2.8 "Storage").

Two stores with one interface:
- GCSStore: the private bucket named by COLLECT_BUCKET (the deployed service,
  and Prem's tools with his application-default credentials).
- LocalStore: a folder on disk, for local runs and tests. Defaults to
  data/collect_dev/ (gitignored, like all of data/).

Every object has a generation number. put_json(..., if_generation=g) only
writes if the object is still at generation g (0 = must not exist yet) and
raises Conflict otherwise, so two open tabs can't overwrite each other's
session.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEFAULT = ROOT / "data" / "collect_dev"


class Conflict(Exception):
    """The object changed (or already exists) since it was read."""


class Store(Protocol):
    def get_bytes(self, path: str) -> tuple[bytes, int] | None: ...
    def put_bytes(self, path: str, data: bytes, content_type: str, if_generation: int | None = None) -> int: ...
    def delete(self, path: str) -> None: ...
    def list(self, prefix: str) -> list[str]: ...


def get_json(store: Store, path: str) -> tuple[dict[str, Any], int] | None:
    got = store.get_bytes(path)
    return None if got is None else (json.loads(got[0]), got[1])


def put_json(store: Store, path: str, obj: Any, if_generation: int | None = None) -> int:
    data = json.dumps(obj, indent=1, ensure_ascii=False).encode("utf-8")
    return store.put_bytes(path, data, "application/json", if_generation)


def create_json(store: Store, path: str, obj: Any) -> int:
    """Write only if the object doesn't exist yet; raises Conflict otherwise."""
    return put_json(store, path, obj, if_generation=0)


class LocalStore:
    def __init__(self, root: str | Path = LOCAL_DEFAULT):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _file(self, path: str) -> Path:
        if path.startswith("/") or ".." in Path(path).parts:
            raise ValueError(f"bad object path {path!r}")
        return self.root / path

    def _gen(self, f: Path) -> int:
        g = f.with_name(f.name + ".gen")
        return int(g.read_text()) if g.exists() else (1 if f.exists() else 0)

    def get_bytes(self, path: str) -> tuple[bytes, int] | None:
        f = self._file(path)
        with self._lock:
            return (f.read_bytes(), self._gen(f)) if f.exists() else None

    def put_bytes(self, path: str, data: bytes, content_type: str, if_generation: int | None = None) -> int:
        f = self._file(path)
        with self._lock:
            cur = self._gen(f) if f.exists() else 0
            if if_generation is not None and cur != if_generation:
                raise Conflict(path)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
            f.with_name(f.name + ".gen").write_text(str(cur + 1))
            return cur + 1

    def delete(self, path: str) -> None:
        f = self._file(path)
        with self._lock:
            f.unlink(missing_ok=True)
            f.with_name(f.name + ".gen").unlink(missing_ok=True)

    def list(self, prefix: str) -> list[str]:
        if not self.root.exists():
            return []
        out = []
        for f in self.root.rglob("*"):
            if f.is_file() and not f.name.endswith(".gen"):
                rel = f.relative_to(self.root).as_posix()
                if rel.startswith(prefix):
                    out.append(rel)
        return sorted(out)


class GCSStore:
    def __init__(self, bucket: str):
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)

    def get_bytes(self, path: str) -> tuple[bytes, int] | None:
        from google.api_core.exceptions import NotFound

        blob = self.bucket.blob(path)
        try:
            data = blob.download_as_bytes()
        except NotFound:
            return None
        return data, int(blob.generation)

    def put_bytes(self, path: str, data: bytes, content_type: str, if_generation: int | None = None) -> int:
        from google.api_core.exceptions import PreconditionFailed

        blob = self.bucket.blob(path)
        try:
            blob.upload_from_string(data, content_type=content_type, if_generation_match=if_generation)
        except PreconditionFailed as e:
            raise Conflict(path) from e
        return int(blob.generation)

    def delete(self, path: str) -> None:
        from google.api_core.exceptions import NotFound

        try:
            self.bucket.blob(path).delete()
        except NotFound:
            pass

    def list(self, prefix: str) -> list[str]:
        return sorted(b.name for b in self.bucket.client.list_blobs(self.bucket, prefix=prefix))


def store_from_env(local: bool = False) -> Store:
    """GCSStore(COLLECT_BUCKET) if set (and local is False), else LocalStore(data/collect_dev)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    bucket = os.environ.get("COLLECT_BUCKET")
    if bucket and not local:
        return GCSStore(bucket)
    return LocalStore(os.environ.get("COLLECT_LOCAL_DIR", LOCAL_DEFAULT))
