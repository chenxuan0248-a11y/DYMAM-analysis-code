"""Stage an editable, no-subfolder LaTeX source *dry run* without altering paper/.

The result remains a proxy `article` manuscript with author-controlled TODOs. It is not
the final Springer submission package until those are resolved and QA is repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
PAPER = EXP / "paper"
ROOT = EXP.parents[1]
COMMAND = re.compile(r"\\(input|includegraphics)(\[[^]]*\])?\{([^}]+)\}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    dest = args.destination.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"refusing nonempty destination: {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    staged: dict[str, str] = {}
    in_progress: set[Path] = set()

    def stage(source: Path, preferred: str | None = None) -> str:
        source = source.resolve()
        if not source.is_file() or not source.is_relative_to(ROOT):
            raise ValueError(f"missing or out-of-workspace input: {source}")
        key = str(source.relative_to(ROOT))
        if key in staged:
            return staged[key]
        if source in in_progress:
            raise ValueError(f"recursive LaTeX include: {source}")
        name = preferred or "__".join(source.relative_to(EXP).parts)
        if (dest / name).exists():
            raise ValueError(f"flat filename collision: {name}")
        staged[key] = name
        if source.suffix.lower() != ".tex":
            shutil.copyfile(source, dest / name)
            return name
        in_progress.add(source)
        content = source.read_text()

        def replace(match: re.Match) -> str:
            kind, opts, item = match.groups()
            target = source.parent / item
            if kind == "input" and not target.suffix:
                target = target.with_suffix(".tex")
            elif kind == "includegraphics" and not target.suffix:
                raise ValueError(f"graphics extension required: {item}")
            if kind == "includegraphics" and not target.is_file():
                target = EXP / item  # main.tex uses \\graphicspath{{../}}
            flat = stage(target)
            return f"\\{kind}{opts or ''}{{{flat}}}"

        staged_content = COMMAND.sub(replace, content)
        if source == PAPER / "main.tex":
            staged_content = staged_content.replace(r"\graphicspath{{../}}", r"\graphicspath{{./}}")
        (dest / name).write_text(staged_content)
        in_progress.remove(source)
        return name

    stage(PAPER / "main.tex", "main.tex")
    stage(PAPER / "refs.bib", "refs.bib")
    manifest = {source: {"flat_name": flat,
                         "sha256": hashlib.sha256((dest / flat).read_bytes()).hexdigest()}
                for source, flat in sorted(staged.items())}
    (dest / "SOURCE_MAP.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Staged {len(manifest)} editable/figure files in {dest}; no source file changed")


if __name__ == "__main__":
    main()
