import unicodedata


def norm(s: str) -> str:
    """Lower-case, accent-free, punctuation-free form of a name for matching."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return " ".join(s.replace(".", " ").replace("-", " ").split())


def find_by_name(items, name: str, label: str, get=lambda x: x.name):
    """Unique case/accent-insensitive substring match; exits with a helpful message otherwise."""
    q = norm(name)
    hits = [x for x in items if q in norm(get(x))]
    exact = [x for x in hits if norm(get(x)) == q]
    if len(exact) == 1:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(sorted(get(x) for x in items))
    if not hits:
        raise SystemExit(f"Could not find '{name}' among {label}. Available: {names}")
    raise SystemExit(f"'{name}' matches several players in {label} ({', '.join(get(x) for x in hits)}) - be more specific.")