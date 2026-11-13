"""Telecharge les documents publics listes dans data/sources.json vers data/raw/.

Aucune donnee n'est versionnee dans le depot : ce script les recupere.
Chaque fichier est telecharge avec son statut HTTP et son SHA-256 ecrits dans
data/raw/MANIFEST.json, pour que le corpus utilise soit tracable.

Verifiez la licence/les conditions de reutilisation de chaque source avant de
redistribuer les fichiers. Ajoutez vos propres URL (PDF et surtout DOCX) dans
data/sources.json : l'evaluation d'une strategie "par clause" demande au moins
un DOCX.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SOURCES = ROOT / "sources.json"


def download(url: str, dest: Path, timeout: int = 60) -> tuple[int | None, str | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "rag-hybride-docling/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:  # reseau, DNS, proxy...
        print(f"  erreur reseau: {e}", file=sys.stderr)
        return None, None
    dest.write_bytes(data)
    return status, hashlib.sha256(data).hexdigest()


def main() -> int:
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    RAW.mkdir(parents=True, exist_ok=True)
    manifest, failures = [], 0
    for s in sources:
        dest = RAW / s["filename"]
        print(f"- {s['filename']} <- {s['url']}")
        status, sha = download(s["url"], dest)
        ok = status == 200 and sha is not None
        failures += not ok
        manifest.append({**s, "http_status": status, "sha256": sha, "downloaded": ok})
        print(f"  {'OK' if ok else 'ECHEC'} (HTTP {status})")
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(manifest) - failures}/{len(manifest)} fichiers telecharges.")
    return 1 if failures == len(manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
