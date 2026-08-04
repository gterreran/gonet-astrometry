#!/usr/bin/env bash
set -euo pipefail

visibility="${1:-private}"
if [[ "$visibility" != "private" && "$visibility" != "public" ]]; then
    printf 'Usage: %s [private|public]\n' "$0" >&2
    exit 2
fi

if [[ -d .git ]]; then
    printf 'A Git repository already exists in %s\n' "$PWD" >&2
    exit 1
fi

command -v git >/dev/null || {
    printf 'git is required\n' >&2
    exit 1
}
command -v gh >/dev/null || {
    printf 'GitHub CLI (gh) is required\n' >&2
    exit 1
}

git init -b main
git add .
git commit -m "Initialize GONet Astrometry project"

gh repo create gonet-astrometry \
    "--${visibility}" \
    --source=. \
    --remote=origin \
    --description="Astrometric calibration of wide-field GONet images" \
    --push

git branch dev
git push --set-upstream origin dev
git switch dev

printf '\nRepository initialized. Active branch: dev\n'
