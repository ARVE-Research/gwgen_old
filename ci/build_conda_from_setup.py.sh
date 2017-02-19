#!/bin/bash

DIR=$1

cd $DIR
# build the tarball
python setup.py sdist
FNAME=`ls -rt dist/* | tail -n 1`
# create the receipt
mkdir tmp
conda skeleton pypi --output-dir tmp --manual-url file://`pwd`/${FNAME}
# build
cd tmp
conda build `ls .`
# print the output
conda build `ls .` --output
cd ../
rm -rf tmp
