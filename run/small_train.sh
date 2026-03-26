#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/.."
exec ./run/train_26m.sh "$@"
