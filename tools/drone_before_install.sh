#!/usr/bin/env bash
set -ex

export WHEELHOUSE="--no-index --trusted-host travis-wheels.scikit-image.org --find-links=http://travis-wheels.scikit-image.org/"
export PIP_DEFAULT_TIMEOUT=60
sudo service xvfb start
export DISPLAY=:99.0
export PYTHONWARNINGS="all"
export TEST_ARGS="--exe --ignore-files=^_test -v --with-doctest --ignore-files=^setup.py$"


retry () {
    # https://gist.github.com/fungusakafungus/1026804
    local retry_max=3
    local count=$retry_max
    while [ $count -gt 0 ]; do
        "$@" && break
        count=$(($count - 1))
        sleep 1
    done

    [ $count -eq 0 ] && {
        echo "Retry failed [$retry_max]: $@" >&2
        return 1
    }
    return 0
}

sudo apt-get install python3-numpy python3-scipy cython \
                     python3-six python3-tk

export PYTHONWARNINGS="ignore"

virtualenv -p python --system-site-packages ~/venv
source ~/venv/bin/activate

retry pip install --upgrade setuptools pip
retry pip install wheel flake8 coveralls nose

retry pip install $WHEELHOUSE numpy
retry pip install dask[array]
retry pip install $WHEELHOUSE -r requirements.txt
retry pip install $WHEELHOUSE cython

export PYTHONWARNINGS="default"
export retry

set +ex
