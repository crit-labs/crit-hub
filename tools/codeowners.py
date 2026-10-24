"""Who approves a change: .github/CODEOWNERS is rendered from the engine cards.

Every engine folder belongs to its contributor: engines/<owner>/ is owned by the GitHub account
<owner>, the contributor.name its cards give, and a change to it needs that account's approval.
Everything else (the index, the template, schemas, tools, CI, .github) is owned by the crit-labs
maintainers, who are also the only ones who merge. Engines contributed by the crit-labs
organization itself are the maintainers' too.

    python -B tools/codeowners.py            write .github/CODEOWNERS from engines/*/*/card.yaml
    python -B tools/codeowners.py --check    fail if .github/CODEOWNERS is out of date
"""
import argparse
from pathlib import Path

import yaml

MAINTAINERS = "@crit-labs/maintainers"
ORGANIZATION = "crit-labs"
PATH = Path(".github") / "CODEOWNERS"
HEADER = """\
# Rendered by tools/codeowners.py from the engine cards: never edit by hand. Run
# python -B tools/codeowners.py after adding an engine under a new account.
#
# Anyone may open a pull request; only the crit-labs maintainers merge. A change to an engine folder
# also needs the approval of the account that contributed it (tools/check_owners.py checks it).
"""


def owner_of(account):
    """The CODEOWNERS owner of engines/<account>/: that account, or the maintainers for crit-labs."""
    return MAINTAINERS if account.lower() == ORGANIZATION else f"@{account}"


def accounts(hub):
    """Every contributor.name the engine cards give, which is also their folder's <owner>."""
    names = set()
    for path in sorted((Path(hub) / "engines").glob("*/*/card.yaml")):
        if path.parent.parent.name.startswith("_"):
            continue
        card = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = card.get("contributor", {}).get("name") if isinstance(card, dict) else None
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return sorted(names, key=str.lower)


def render(hub):
    lines = [HEADER, f"* {MAINTAINERS}", ""]
    lines += [f"/engines/{name}/ {owner_of(name)}" for name in accounts(hub)]
    return "\n".join(lines) + "\n"


def main(hub, check):
    path = Path(hub) / PATH
    rendered = render(hub)
    if check:
        if not path.is_file() or path.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{PATH.as_posix()} is not rendered from the engine cards; run python -B tools/codeowners.py")
        print(f"{PATH.as_posix()} matches the engine cards")
        return
    path.parent.mkdir(exist_ok=True)
    path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {PATH.as_posix()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hub", default=".", help="crit-hub checkout (default: the current folder)")
    p.add_argument("--check", action="store_true", help="fail if .github/CODEOWNERS differs from the rendered one")
    args = p.parse_args()
    main(args.hub, args.check)
