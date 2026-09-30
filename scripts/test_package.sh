#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

usage() {
	cat <<'EOF2'
Install this package's current version from the index into a fresh venv
(outside the repo) and import it.

Usage: scripts/test_package.sh [options]

Options:
  --testpypi   Install from https://test.pypi.org (only this package and mcp-nav-shared; the rest from PyPI)
  -h, --help   Show this help
EOF2
}

testpypi=false
while [[ $# -gt 0 ]]; do
	case "$1" in
	--testpypi)
		testpypi=true
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "error: unknown option: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

read -r name version < <(python3 -c "
import tomllib
p = tomllib.load(open('${repo_root}/pyproject.toml', 'rb'))['project']
print(p['name'], p['version'])")

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT
uv venv --quiet "${work}/venv"

echo "Installing ${name}==${version}"
if [[ "${testpypi}" == "true" ]]; then
	# Only our own packages come from TestPyPI (--no-deps, so no dependency can be
	# squatted there); their dependencies then resolve from real PyPI below.
	own=("${name}==${version}")
	shared="$(python3 -c "
import tomllib
deps = tomllib.load(open('${repo_root}/pyproject.toml', 'rb'))['project']['dependencies']
print(next((d for d in deps if d.startswith('mcp-nav-shared')), ''))")"
	[[ -n "${shared}" ]] && own+=("${shared}")
	uv pip install --python "${work}/venv" --no-deps --refresh --index-strategy unsafe-best-match \
		--default-index https://test.pypi.org/simple/ "${own[@]}"
	"${work}/venv/bin/python" -c "
import importlib.metadata as m
print('\\n'.join(r for r in (m.requires('${name}') or []) if 'extra ==' not in r))" |
		uv pip install --python "${work}/venv" -r -
else
	uv pip install --refresh --python "${work}/venv" "${name}==${version}"
fi

"${work}/venv/bin/python" -c "import mcp_nav_shared, mcp_nav_shared.lsp_client, mcp_nav_shared.format; print('import ok')"
echo "OK"
