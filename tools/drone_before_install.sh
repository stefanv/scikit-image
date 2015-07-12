#!/usr/bin/env bash
set -ex

export WHEELHOUSE="--no-index --find-links=http://travis-wheels.scikit-image.org/"
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
        cat /home/ubuntu/.pip/pip.log || echo "No pip log"
        count=$(($count - 1))
        sleep 1
    done

    [ $count -eq 0 ] && {
        echo "Retry failed [$retry_max]: $@" >&2
        return 1
    }
    return 0
}

sudo apt-get install python-numpy python-scipy python-networkx cython \
                     python-six python-tk

virtualenv -p python --system-site-packages ~/venv

source ~/venv/bin/activate
retry pip install --upgrade setuptools pip
python -c 'import setuptools; print("Setuptools version:", setuptools.__version__)'
retry pip install wheel flake8 coveralls nose
retry pip install $WHEELHOUSE -r requirements.txt
# clean up disk space
sudo apt-get clean
sudo rm -rf /tmp/*

export retry

set +ex

