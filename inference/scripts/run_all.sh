#!/bin/bash
# One-click: start servers + run inference
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$RELEASE_DIR"

# Load .env if exists
if [ -f .env ]; then
    set -a; source .env; set +a
    echo "✓ Loaded .env"
fi

echo "=========================================="
echo "  LiteResearcher - Full Pipeline"
echo "=========================================="

# Check required env vars
if [ -z "$MODEL" ]; then
    echo "Error: MODEL not set. Set MODEL env var or add to .env"
    exit 1
fi

if [ -z "$SERPER_KEY_ID" ]; then
    echo "Warning: SERPER_KEY_ID not set. Search will fail."
fi

if [ "${BROWSER_PROVIDER:-jina}" = "scrapedo" ] && [ -z "$SCRAPEDO_API_KEY" ]; then
    echo "Warning: BROWSER_PROVIDER=scrapedo but SCRAPEDO_API_KEY not set. Browse will fail."
fi

# Start servers in background
echo ""
echo "Step 1/2: Starting servers..."
echo "→ Search server on port ${SEARCH_SERVER_PORT:-8001}"
python src/search_server.py &
SEARCH_PID=$!

echo "→ Browser server on port ${BROWSER_SERVER_PORT:-8002}"
python src/browser_server.py &
BROWSER_PID=$!

# Wait for servers to be ready
echo "  Waiting for servers..."
sleep 3

# Check health
curl -sf "http://127.0.0.1:${SEARCH_SERVER_PORT:-8001}/health" > /dev/null 2>&1 && echo "  ✓ Search server ready" || echo "  ⚠ Search server not ready"
curl -sf "http://127.0.0.1:${BROWSER_SERVER_PORT:-8002}/health" > /dev/null 2>&1 && echo "  ✓ Browser server ready" || echo "  ⚠ Browser server not ready"

# Trap to clean up
trap "kill $SEARCH_PID $BROWSER_PID 2>/dev/null; echo ''; echo 'Servers stopped.'" EXIT

# Run inference
echo ""
echo "Step 2/2: Running inference..."
bash "$SCRIPT_DIR/run_inference.sh" "$@"

echo ""
echo "✓ Pipeline complete"
