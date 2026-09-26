#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Zaheer Sheriff K
"""Replace the absolute paths a record's tool stamped with placeholders, so it can be committed.

    python3 scripts/scrub_record_paths.py DIR --prefix NAME=/abs/path [--prefix ...]

``DIR`` holds gzipped JSON records and the ``MANIFEST.json`` that lists them (the layout of
``rows/exploratory/step1-a6000-2026-09-26/``). For every record the manifest lists:

1. the gz file and its decompressed bytes must have the manifest's ``sha256_gz`` and
   ``sha256_raw``, or nothing is written (exit 2);
2. each ``--prefix`` path is replaced by ``<NAME>`` wherever it stands in the decompressed text
   as a whole path or the head of one (``/a/b`` in ``/a/b/c`` or ``/a/b:``, never in ``/a/bc``),
   longest prefix first; the rest of the bytes are left as they are;
3. a record that names another record here by its ``sha256_raw`` is rewritten to name that
   record's new sha256 (a places record names its captures so), until no link changes;
4. what remains must hold no machine path (``machine_paths``): an absolute path that is not a
   placeholder's tail or the kernel's own ``/proc/``, ``/dev/`` or ``/sys/``. One left is a
   prefix not given, and nothing is written (exit 2);
5. each changed record is compressed again with ``gzip -9 -n`` and its manifest entry gets the
   new ``sha256_raw``, ``bytes_raw``, ``sha256_gz`` and ``bytes_gz``, keeping the record as
   its tool wrote it as ``sha256_as_written`` and ``bytes_as_written``, and what was replaced
   as ``scrub`` (placeholders by count, and links rewritten). A record with nothing to replace
   is left byte for byte; running the scrub again changes nothing.

The placeholder names are the ones ``PLACEHOLDERS`` defines, so a scrubbed record says what each
stood for without saying where it was. The prefixes are given on the command line and are
written nowhere. Anyone holding the records as written reruns this with the same prefixes and
gets the same bytes, which ``sha256_as_written`` lets them check. Nothing here touches a GPU.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MANIFEST = "MANIFEST.json"
EXIT_OK, EXIT_REFUSED = 0, 2

#: What each placeholder stands for; ``--prefix`` names one of these.
PLACEHOLDERS: dict[str, str] = {
    "checkout": "the verbatim checkout the tool and the server ran from",
    "venv": "the Python virtual environment they ran in",
    "home": "the home directory of the user who ran them",
    "step1-out": "the directory the step-1 tools wrote their records and logs to",
}

#: An absolute path: a slash that does not follow a word, a dot, a tilde, a placeholder's
#: closing bracket or another slash (a URL's "//host/..."), then at least one directory. The
#: kernel's own /proc/, /dev/ and /sys/ name nothing about a machine and are allowed.
MACHINE_PATH = re.compile(r"(?<![\w.~<>/])/(?!(?:proc|dev|sys)/)(?:[\w.-]+/)+[\w.-]*")
_SHA256 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")


class Refused(ValueError):
    """Nothing is written; ``main`` prints why and exits ``EXIT_REFUSED``."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def machine_paths(node: Any, where: str = "") -> list[str]:
    """Every string in ``node`` that holds an absolute path, with where it is."""
    if isinstance(node, Mapping):
        return [p for key, value in node.items() for p in machine_paths(value, f"{where}/{key}")]
    if isinstance(node, list | tuple):
        return [p for i, value in enumerate(node) for p in machine_paths(value, f"{where}[{i}]")]
    if isinstance(node, str) and MACHINE_PATH.search(node):
        return [f"{where}: {node[:120]!r}"]
    return []


def parse_prefixes(given: Sequence[str]) -> dict[str, str]:
    """``NAME=/abs/path`` pairs as {path: "<NAME>"}; refused unless NAME is a placeholder and
    the path absolute, with no trailing slash, and not given twice."""
    out: dict[str, str] = {}
    for item in given:
        name, sep, path = item.partition("=")
        if not sep or name not in PLACEHOLDERS:
            raise Refused(
                f"--prefix {item!r}: expected NAME=/abs/path, NAME one of {sorted(PLACEHOLDERS)}"
            )
        if not path.startswith("/") or path.endswith("/") or len(path) < 2:
            raise Refused(f"--prefix {item!r}: the path must be absolute, without a trailing slash")
        if path in out:
            raise Refused(f"--prefix {item!r}: that path is given twice")
        out[path] = f"<{name}>"
    return out


def replace_prefixes(text: str, prefixes: Mapping[str, str]) -> tuple[str, dict[str, int]]:
    """``text`` with every prefix replaced by its placeholder, longest first, and the count of
    each placeholder put in. A prefix is replaced only where the path ends or goes on with a
    slash, never inside a longer name."""
    counts: dict[str, int] = {}
    for path in sorted(prefixes, key=len, reverse=True):
        pattern = re.compile(re.escape(path) + r"(?![\w.-])")
        text, n = pattern.subn(prefixes[path].replace("\\", "\\\\"), text)
        if n:
            name = prefixes[path][1:-1]
            counts[name] = counts.get(name, 0) + n
    return text, counts


def relink(text: str, moves: Mapping[str, str]) -> tuple[str, int]:
    """``text`` with every whole sha256 that ``moves`` maps replaced, and how many were."""
    found = [m.group(0) for m in _SHA256.finditer(text) if m.group(0) in moves]
    return _SHA256.sub(lambda m: moves.get(m.group(0), m.group(0)), text), len(found)


def gzip_n9(raw: bytes) -> bytes:
    """``gzip -9 -n`` of ``raw``: no name or time in the header, as the manifest states."""
    if shutil.which("gzip") is None:
        raise Refused("gzip is not on PATH; the manifest's files are gzip -9 -n")
    done = subprocess.run(["gzip", "-9", "-n", "-c"], input=raw, capture_output=True, check=True)
    if gzip.decompress(done.stdout) != raw:
        raise Refused("gzip -9 -n did not give back the bytes it was given")
    return done.stdout


def read_entry(directory: Path, name: str, entry: Mapping[str, Any]) -> bytes:
    packed = (directory / name).read_bytes()
    if sha256(packed) != entry["sha256_gz"]:
        raise Refused(
            f"{name}: its sha256 is {sha256(packed)}, the manifest says {entry['sha256_gz']}"
        )
    raw = gzip.decompress(packed)
    if sha256(raw) != entry["sha256_raw"]:
        raise Refused(
            f"{name}: the sha256 of its decompressed bytes is {sha256(raw)}, the manifest says "
            f"{entry['sha256_raw']}"
        )
    return raw


def scrub(directory: Path, prefixes: Mapping[str, str]) -> dict[str, Any]:
    """The scrubbed records and manifest for ``directory``, computed in memory and refused before
    anything is written. Returns {"manifest": dict, "files": {name: gz bytes to write}}."""
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    entries: dict[str, dict[str, Any]] = manifest["records"]
    listed = set(entries)
    present = {p.name for p in directory.glob("*.json.gz")}
    if listed != present:
        raise Refused(
            f"{directory.name}: the manifest lists {sorted(listed - present)} that are not here "
            f"and not {sorted(present - listed)} that are"
        )
    before = {name: read_entry(directory, name, entry) for name, entry in sorted(entries.items())}
    texts: dict[str, str] = {}
    placed: dict[str, dict[str, int]] = {}
    for name, raw in before.items():
        texts[name], placed[name] = replace_prefixes(raw.decode("utf-8"), prefixes)
    # Links: a record naming another by its sha256 names the scrubbed one. ``held`` is the
    # sha256 by which the others name each record now; a rewrite changes the linking record's
    # own sha256, so repeat until nothing moves (a cycle of links never settles, and is refused).
    relinked = {name: 0 for name in texts}
    held = {name: entries[name]["sha256_raw"] for name in texts}
    for _ in range(len(texts) + 1):
        moves = {
            held[name]: sha256(text.encode("utf-8"))
            for name, text in texts.items()
            if sha256(text.encode("utf-8")) != held[name]
        }
        if not moves:
            break
        for name, text in texts.items():
            texts[name], count = relink(text, moves)
            relinked[name] += count
        held = {name: moves.get(held[name], held[name]) for name in texts}
    else:
        raise Refused("the sha256 links between the records do not settle")
    files: dict[str, bytes] = {}
    out = json.loads(json.dumps(manifest))
    for name, text in texts.items():
        record = json.loads(text)
        left = machine_paths(record)
        if left:
            raise Refused(f"{name} still holds a machine path: " + "; ".join(left[:5]))
        raw = text.encode("utf-8")
        if raw == before[name]:
            continue
        packed = gzip_n9(raw)
        entry = out["records"][name]
        entry.setdefault("sha256_as_written", entry["sha256_raw"])
        entry.setdefault("bytes_as_written", entry["bytes_raw"])
        entry["sha256_raw"], entry["bytes_raw"] = sha256(raw), len(raw)
        entry["sha256_gz"], entry["bytes_gz"] = sha256(packed), len(packed)
        scrubbed = entry.setdefault("scrub", {"placeholders": {}, "links_rewritten": 0})
        for key, n in placed[name].items():
            scrubbed["placeholders"][key] = scrubbed["placeholders"].get(key, 0) + n
        scrubbed["links_rewritten"] += relinked[name]
        files[name] = packed
    if files:
        out["scrubbed"] = (
            "scripts/scrub_record_paths.py replaced the absolute paths each record's tool "
            "stamped by a placeholder: "
            + "; ".join(f"<{k}>, {v}" for k, v in sorted(PLACEHOLDERS.items()))
            + ". A record that names another here by sha256 names the scrubbed one. "
            "sha256_as_written and bytes_as_written are the record as its tool wrote it; scrub "
            "says what was replaced. A record without them held no path and is as written."
        )
    return {"manifest": out, "files": files}


def write(directory: Path, result: Mapping[str, Any]) -> None:
    """Each changed gz file, then the manifest, each through a temporary file and a rename."""
    for name, packed in result["files"].items():
        tmp = directory / f".{name}.tmp"
        tmp.write_bytes(packed)
        os.replace(tmp, directory / name)
    tmp = directory / f".{MANIFEST}.tmp"
    tmp.write_text(
        json.dumps(result["manifest"], indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(tmp, directory / MANIFEST)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", action="append", default=[], metavar="NAME=/abs/path")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = scrub(args.directory, parse_prefixes(args.prefix))
    except (Refused, OSError, ValueError, KeyError) as exc:
        print(f"scrub_record_paths: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    write(args.directory, result)
    for name in sorted(result["files"]):
        entry = result["manifest"]["records"][name]
        print(
            f"{name}: {entry['scrub']['placeholders']}, links {entry['scrub']['links_rewritten']}"
        )
    print(f"scrub_record_paths: {len(result['files'])} records rewritten")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
