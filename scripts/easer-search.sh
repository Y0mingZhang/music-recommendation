#!/bin/bash

set -Eeuo pipefail

for lambda in 1e-1 1e0 1e1 1e2 1e3 1e4
do
    python src/recommend/easer.py --lambda_ $lambda > outputs/$lambda.txt
done