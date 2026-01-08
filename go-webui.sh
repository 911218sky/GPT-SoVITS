#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$SCRIPT_DIR/runtime/python/bin:$PATH"
./runtime/python/bin/python webui.py
