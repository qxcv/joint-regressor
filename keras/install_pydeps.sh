#!/bin/bash

set -e

ENV_DIR=env/

_reset() {
    echo -en "\033[0m"
}

_green() {
    echo -en "\033[1;32m"
}

_red() {
    echo -en "\033[1;31m"
}

out() {
    _green; echo $*; _reset;
}

error() {
    _red; echo $*; _reset;
}

if [ ! -d "$ENV_DIR" ]; then
    out "Creating virtualenv"
    # Unfortunately we have to use Python 2 because there are no Python 3
    # development libraries installed on paloalto. It's probably that numpy,
    # scipy et al. are not installed for Py3K either.
    virtualenv --system-site-packages -p "$(which python2)" "$ENV_DIR"
elif [ "$1" != "--force" ]; then
    error "Already have virtualenv. Check that you're not overwriting anything"
    exit
fi

source "$ENV_DIR"/bin/activate
pip install -r requirements.txt
deactivate
out "Done"
