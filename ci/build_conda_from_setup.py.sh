#!/bin/bash

function build_conda() {

    DIR=$1

    WORK=`pwd`

    cd $DIR
    # build the tarball
    python setup.py sdist
    FNAME=`ls -rt dist/* | tail -n 1`
    # create the receipt
    mkdir tmp
    RUNNING_SKELETON=1 conda skeleton pypi --output-dir tmp --python-version ${PYTHON_VERSION} \
        --manual-url file://`pwd`/${FNAME}
    # build
    cd tmp
    conda build --python ${PYTHON_VERSION} `ls .`
    # print the output
    BUILD_FNAME=$(conda build --python ${PYTHON_VERSION} `ls .` --output)
    cd ../
    rm -rf tmp
    cd $WORK
}