#!/bin/bash
cd $1
python ~/projects/pylearn2/pylearn2/scripts/train.py model.yaml || exit -1
