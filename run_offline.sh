#!/bin/zsh -f
# macOS launcher: source files are readable, generated files stay in artifacts/,
# network is denied. Options come before the command:
#   zsh run_offline.sh [--python PY] [--runtime DIR] [--data-root DIR] COMMAND ...
set -euo pipefail
TOOLKIT_ROOT="${0:A:h}"
TOOLKIT_PYTHON="$(command -v python3 || true)"
TOOLKIT_RUNTIME=''
TOOLKIT_DATA=''
while [[ "${1:-}" == '--python' || "${1:-}" == '--runtime' || "${1:-}" == '--data-root' ]]; do
  [[ $# -ge 2 ]] || { print -u2 'Missing option value'; exit 2; }
  case "$1" in
    --python) TOOLKIT_PYTHON="$2" ;;
    --runtime) TOOLKIT_RUNTIME="$2" ;;
    --data-root) TOOLKIT_DATA="$2" ;;
  esac
  shift 2
done
[[ -n "$TOOLKIT_PYTHON" && -x "$TOOLKIT_PYTHON" ]] || { print -u2 'Python 3.11+ is required; pass --python /absolute/path/to/python3'; exit 2; }
TOOLKIT_PYTHON="${TOOLKIT_PYTHON:A}"
[[ -n "$TOOLKIT_RUNTIME" ]] || TOOLKIT_RUNTIME="${TOOLKIT_PYTHON:h:h}"
TOOLKIT_RUNTIME="${TOOLKIT_RUNTIME:A}"
[[ -d "$TOOLKIT_RUNTIME" && "$TOOLKIT_RUNTIME" != '/' && "$TOOLKIT_RUNTIME" != '/Users' && "$TOOLKIT_RUNTIME" != "${HOME:-}" ]] || { print -u2 'Select a dedicated Python runtime directory'; exit 2; }
[[ -x /usr/bin/sandbox-exec ]] || { print -u2 'This launcher requires macOS sandbox-exec; no unisolated fallback is provided.'; exit 2; }
TOOLKIT_SSL="$TOOLKIT_RUNTIME"
TOOLKIT_DECIMAL="$TOOLKIT_RUNTIME"
case "$TOOLKIT_PYTHON" in
  /opt/homebrew/Cellar/*) TOOLKIT_SSL=/opt/homebrew/Cellar/openssl@3; TOOLKIT_DECIMAL=/opt/homebrew/Cellar/mpdecimal ;;
  /usr/local/Cellar/*) TOOLKIT_SSL=/usr/local/Cellar/openssl@3; TOOLKIT_DECIMAL=/usr/local/Cellar/mpdecimal ;;
esac
TOOLKIT_PROFILE="$(<"$TOOLKIT_ROOT/offline.sb")"
if [[ -n "$TOOLKIT_DATA" ]]; then
  # Grant read-only access to one explicitly named local media directory.
  TOOLKIT_DATA="${TOOLKIT_DATA:A}"
  if [[ ! -d "$TOOLKIT_DATA" || "$TOOLKIT_DATA" == / || "$TOOLKIT_DATA" == */.* \
        || "$TOOLKIT_DATA" == *[\"\\]* || "$TOOLKIT_DATA" == *$'\n'* ]]; then
    print -u2 "run_offline.sh: refusing data root (missing, /, hidden path or unsafe characters): $TOOLKIT_DATA"
    exit 2
  fi
  if [[ "$TOOLKIT_ROOT/" == "$TOOLKIT_DATA"/* || "${HOME:-/}/" == "$TOOLKIT_DATA"/* ]]; then
    print -u2 "run_offline.sh: refusing a data root that contains the package or home directory: $TOOLKIT_DATA"
    exit 2
  fi
  TOOLKIT_PROFILE+=$'\n'"(allow file-read* (subpath \"$TOOLKIT_DATA\"))"
fi
mkdir -p "$TOOLKIT_ROOT/artifacts/tmp"
ulimit -t 180
ulimit -f 131072
cd "$TOOLKIT_ROOT"
exec /usr/bin/env -i PATH=/usr/bin:/bin TMPDIR="$TOOLKIT_ROOT/artifacts/tmp" /usr/bin/sandbox-exec \
  -D "PYTHON=$TOOLKIT_PYTHON" -D "RUNTIME=$TOOLKIT_RUNTIME" \
  -D "PYTHON_APP=${TOOLKIT_PYTHON:h:h}/Resources/Python.app/Contents/MacOS/Python" \
  -D "SSL=$TOOLKIT_SSL" -D "DECIMAL=$TOOLKIT_DECIMAL" \
  -D "PACKAGE=$TOOLKIT_ROOT" -D "WRITABLE=$TOOLKIT_ROOT/artifacts" \
  -p "$TOOLKIT_PROFILE" "$TOOLKIT_PYTHON" -I -B "$TOOLKIT_ROOT/offline_entry.py" "$@"
