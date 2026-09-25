#!/bin/bash
# sources the .env file and runs ui.ipynb via voila — the SEPAL app entry path.
# Open http://127.0.0.1:PORT/voila/render/ui.ipynb
# Usage: ./run_ui.sh [--port PORT]
# If no port is provided, defaults to 8911 (run_solara.sh uses 8910)

PORT="8911"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#--port=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--port PORT]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Run from the script's own directory so relative paths resolve
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Make the module root importable (so `import gui...` works)
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ -f .env ]]; then
  while IFS= read -r line || [[ -n $line ]]; do
    [[ $line =~ ^#.*$ || -z $line ]] && continue

    if [[ $line =~ ^([^=]+)=(.*)$ ]]; then
      name="${BASH_REMATCH[1]}"
      value="${BASH_REMATCH[2]}"

      # Remove quotes if present
      value="${value#\'}"
      value="${value%\'}"
      value="${value#\"}"
      value="${value%\"}"

      export "$name=$value"
    fi
  done < .env
else
  echo "Note: no .env file found in $SCRIPT_DIR, skipping env load."
fi

# GDAL writes its temp files to the CWD when CPL_TMPDIR/TMPDIR/TEMP are unset,
# and on SEPAL the CWD is the read-only shared module mount -- so point it at a
# writable per-user scratch dir. Set after .env so an operator value wins; the
# uid suffix keeps us off another user's directory on a shared /tmp.
export CPL_TMPDIR="${CPL_TMPDIR:-${TMPDIR:-/tmp}/spatial_risk_gdal_$(id -u)}"
mkdir -p "$CPL_TMPDIR"

# Serve voila from inside jupyter-server, as SEPAL does, and route map tiles
# through jupyter-server-proxy's /proxy/{port} like SEPAL's
# LOCALTILESERVER_CLIENT_PREFIX. Standalone `voila ui.ipynb` loads no server
# extensions, so that route would 404 there. Local only: no token, loopback bind.
export LOCALTILESERVER_CLIENT_PREFIX="${LOCALTILESERVER_CLIENT_PREFIX:-/proxy/{port}}"
jupyter server --port="$PORT" --no-browser \
  --ServerApp.ip=127.0.0.1 --IdentityProvider.token= \
  --ServerApp.root_dir="$SCRIPT_DIR" \
  --ServerApp.default_url=/voila/render/ui.ipynb
