#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor
if ! test -d vendor/frontier/.git; then
  git clone --filter=blob:none --no-checkout https://github.com/Einsia/Frontier-Engineering.git vendor/frontier
  git -C vendor/frontier sparse-checkout set benchmarks/JobShop frontier_eval
fi
git -C vendor/frontier checkout e3fa29c193356af2ce1ec8b3d23ab1a2e2410071
docker pull python@sha256:90744cff8f32887f075c47d747a173ff333e9e98801667af93c357fa9f5e28ff
