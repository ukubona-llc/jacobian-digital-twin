#!/usr/bin/env bash
set -e

echo "=== GitHub Pages Bootstrap ==="
read -p "GitHub username: " GH_USER
read -p "Repository name: " GH_REPO
read -p "Private? (y/N): " PRIVATE_INPUT
read -s -p "Personal Access Token: " GH_TOKEN
echo

IS_PRIVATE="false"
[[ "$PRIVATE_INPUT" =~ ^[Yy]$ ]] && IS_PRIVATE="true"

API="https://api.github.com"
REPO_API="$API/repos/$GH_USER/$GH_REPO"

curl -s -o /dev/null -w "%{http_code}" \
  -H "Authorization: token $GH_TOKEN" "$REPO_API" | grep -q 200 || \
curl -s -X POST "$API/user/repos" \
  -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d "{\"name\":\"$GH_REPO\",\"private\":$IS_PRIVATE,\"auto_init\":false}" >/dev/null

mkdir -p "$GH_REPO" && cd "$GH_REPO"
git init && git checkout -B ukhona
echo "# $GH_REPO" > index.md
git add index.md && git commit -m "bootstrap gh-pages"
git remote remove origin 2>/dev/null || true
git remote add origin "https://$GH_USER:$GH_TOKEN@github.com/$GH_USER/$GH_REPO.git"
git push -f origin ukhona

curl -s -X POST "$REPO_API/pages" \
  -H "Authorization: token $GH_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -d '{"source":{"branch":"ukhona","path":"/"}}' >/dev/null || true

echo "======================================"
echo "LIVE (may take ~30s):"
echo "https://$GH_USER.github.io/$GH_REPO/"
echo "======================================"