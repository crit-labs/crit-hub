"""Check that a pull request has the approval of every engine contributor whose folder it changes.

A change to engines/<owner>/ (adding, editing, moving or deleting a file) needs <owner>'s approval:
either <owner> opened the pull request, or <owner>'s latest review approves its latest commit. The
folder names its contributor, so a new account's first engine is opened by that account or approved
by it. Folders the crit-labs maintainers own (engines/crit-labs/, the template) and every shared file
need no further check here: only maintainers can merge, and the branch rules ask for their approval.

GitHub Actions runs this on every pull request and review (.github/workflows/owners.yml):

    python -B tools/check_owners.py --repo crit-labs/crit-hub --pr 42    (GITHUB_TOKEN to read the PR)
"""
import argparse
import json
import os
import urllib.request

import codeowners

API = "https://api.github.com"


def get(path, token):
    """Every page of a GitHub API list, or the one object."""
    items, url = [], f"{API}{path}"
    while url:
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                       "X-GitHub-Api-Version": "2022-11-28",
                                                       **({"Authorization": f"Bearer {token}"} if token else {})})
        with urllib.request.urlopen(request) as response:
            data = json.load(response)
            links = response.headers.get("Link") or ""
        if not isinstance(data, list):
            return data
        items += data
        url = next((part.split(";")[0].strip(" <>") for part in links.split(",") if 'rel="next"' in part), None)
    return items


def owners(paths):
    """The individual accounts whose engine folders these paths are in (maintainer-owned ones left out)."""
    found = {}
    for path in paths:
        parts = path.split("/")
        if len(parts) >= 3 and parts[0] == "engines" and not parts[1].startswith("_") \
                and codeowners.owner_of(parts[1]) != codeowners.MAINTAINERS:
            found.setdefault(parts[1].lower(), parts[1])
    return found


def approvers(reviews, head):
    """Accounts whose latest approving or rejecting review approves the head commit."""
    latest = {}
    for review in reviews:
        if review.get("state") in ("APPROVED", "CHANGES_REQUESTED", "DISMISSED") and review.get("user"):
            latest[review["user"]["login"].lower()] = review
    return {login for login, review in latest.items() if review["state"] == "APPROVED" and review.get("commit_id") == head}


def missing(pull, files, reviews):
    """The owners who neither opened the pull request nor approved its latest commit."""
    paths = [f["filename"] for f in files] + [f["previous_filename"] for f in files if f.get("previous_filename")]
    approved = approvers(reviews, pull["head"]["sha"]) | {pull["user"]["login"].lower()}
    return sorted(name for login, name in owners(paths).items() if login not in approved)


def main(repo, number, token):
    pull = get(f"/repos/{repo}/pulls/{number}", token)
    files = get(f"/repos/{repo}/pulls/{number}/files?per_page=100", token)
    reviews = get(f"/repos/{repo}/pulls/{number}/reviews?per_page=100", token)
    names = missing(pull, files, reviews)
    if names:
        raise SystemExit("This pull request changes engines contributed by " + ", ".join(f"@{n}" for n in names)
                         + ": each of them approves its latest commit before it is merged.")
    print("every engine contributor whose folder this pull request changes opened it or approved it")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="owner/name of the repository")
    p.add_argument("--pr", required=True, type=int, help="pull request number")
    args = p.parse_args()
    main(args.repo, args.pr, os.environ.get("GITHUB_TOKEN"))
