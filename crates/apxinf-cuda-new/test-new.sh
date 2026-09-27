#!/usr/bin/env bash
# Run without replacing the legacy CUDA crate or touching existing families.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export APXINF_CUDA_ARCH="${APXINF_CUDA_ARCH:-sm_110}"
if [[ $# == 0 ]]; then
  cargo test -p apxinf-cuda-next -- --test-threads=1
else
  cargo "$@"
fi
