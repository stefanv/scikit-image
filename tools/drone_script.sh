#!/usr/bin/env bash

set -ex

python -c 'import numpy; print("numpy is:", numpy.__version__)'
pip install -e .
nosetests
