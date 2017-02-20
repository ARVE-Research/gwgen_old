#!/bin/bash

DIR=$1

cd $DIR
# build the tarball
python setup.py sdist
FNAME=`ls -rt dist/* | tail -n 1`
# create the receipt
mkdir tmp
conda skeleton pypi --output-dir tmp --python-version ${PYTHON_VERSION} --manual-url file://`pwd`/${FNAME}
# build
cd tmp
conda build --python ${PYTHON_VERSION} `ls .`
# print the output
conda build --python ${PYTHON_VERSION} `ls .` --output
cd ../
rm -rf tmp
