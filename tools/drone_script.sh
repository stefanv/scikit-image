#!/usr/bin/env bash

set -ex
source ~/venv/bin/activate

./tools/build_versions.py

python -c 'import numpy; print("numpy is:", numpy.__version__)'

pip install -e .
nosetests
