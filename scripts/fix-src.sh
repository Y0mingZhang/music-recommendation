#!/bin/bash

set -Eeuo pipefail

isort --ca --cs --ls --csi src
black src