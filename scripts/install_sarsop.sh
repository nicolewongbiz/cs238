#!/usr/bin/env bash
# Build the SARSOP solver (https://github.com/AdaCompNUS/sarsop) into third_party/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/sarsop"
if [ ! -d "$DEST" ]; then
  git clone --depth 1 https://github.com/AdaCompNUS/sarsop.git "$DEST"
fi
make -C "$DEST/src" -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
echo "Built: $DEST/src/pomdpsol"
