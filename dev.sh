#!/bin/bash
# Runs a command in the project virtualenv. Usage from Windows:
#   wsl -d Ubuntu -e bash dev.sh python -m pytest -q
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source "$HOME/hitl/.venv/bin/activate"
cd "$(dirname "$(readlink -f "$0")")"
exec "$@"
