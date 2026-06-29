#!/bin/bash
# Start search and browser servers
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
echo "  Starting Search & Browser Servers"
echo "=========================================="

# Start search server
echo "→ Starting search server on port ${SEARCH_SERVER_PORT:-8001}..."
python src/search_server.py &
SEARCH_PID=$!
echo "  PID: $SEARCH_PID"

# Start browser server
echo "→ Starting browser server on port ${BROWSER_SERVER_PORT:-8002}..."
python src/browser_server.py &
BROWSER_PID=$!
echo "  PID: $BROWSER_PID"

echo ""
echo "✓ Servers started"
echo "  Search:  http://127.0.0.1:${SEARCH_SERVER_PORT:-8001}/health"
echo "  Browser: http://127.0.0.1:${BROWSER_SERVER_PORT:-8002}/health"
echo ""
echo "Press Ctrl+C to stop all servers"

# Trap to kill both on exit
trap "kill $SEARCH_PID $BROWSER_PID 2>/dev/null; echo 'Servers stopped.'" EXIT

wait
