#!/bin/bash
# Assemble a runnable package: native library, chat adapter, vendored Python deps,
# benchmark/check executable and notices.  make_package.sh SRC BUILD VENDOR OUT
set -euo pipefail
SRC=$1; BUILD=$2; VENDOR=$3; OUT=$4
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/bin" "$OUT/python"
cp "$BUILD/libq38.so" "$OUT/lib/"
cp "$BUILD/q38bench" "$OUT/bin/"
cp "$SRC/python/chat_adapter.py" "$OUT/python/"
cp -r "$VENDOR" "$OUT/vendor"
find "$OUT/vendor" -name "__pycache__" -prune -exec rm -rf {} +
cp "$SRC/THIRD_PARTY.md" "$OUT/" 2>/dev/null || cp "$SRC/../THIRD_PARTY.md" "$OUT/"
