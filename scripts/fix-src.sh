#!/bin/bash

set -Eeuo pipefail

isort --ca --cs --ls --csi src/data/*.py src/recommend/*.py
black src/data/*.py src/recommend/*.py