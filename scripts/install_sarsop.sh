#!/usr/bin/env bash
# Build the SARSOP solver (https://github.com/AdaCompNUS/sarsop) into third_party/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/sarsop"
if [ ! -d "$DEST" ]; then
  git clone --depth 1 https://github.com/AdaCompNUS/sarsop.git "$DEST"
fi
# SARSOP's Makefile hard-codes x86 SSE flags that clang rejects on Apple Silicon/ARM.
if [ "$(uname -m)" != "x86_64" ]; then
  sed -i.bak -e 's/-msse2//g' -e 's/-mfpmath=sse//g' "$DEST/src/Makefile"
fi
# Newer libc++ (macOS) forbids constructing vector iterators from NULL.
sed -i.bak 's/_begin(NULL), _end(NULL)/_begin(), _end()/' "$DEST/src/MathLib/SparseMatrix.h"
# The bundled Cassandra parser is old K&R-style C; clang >= 16 turns these warnings into errors.
make -C "$DEST/src" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" \
  CC="${CC:-gcc} -Wno-error=implicit-function-declaration -Wno-error=implicit-int -Wno-error=int-conversion"
echo "Built: $DEST/src/pomdpsol"
