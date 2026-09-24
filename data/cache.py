import json
import time
from pathlib import Path

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def cached(key: str, fetch, ttl_hours: float = 12):
    """Return JSON-serialisable data from disk if fresh, otherwise call fetch() and store it."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < ttl_hours * 3600:
        return json.loads(path.read_text())
    data = fetch()
    path.write_text(json.dumps(data))
    return data