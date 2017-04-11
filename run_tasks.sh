#!/usr/bin/env sh

python create_lmdb_1.py

~/loc/caffe/build/tools/caffe train -solver quick_solver_all.prototxt -weights ../f1/bvlc_googlenet/bvlc_googlenet.caffemodel 2>&1 | tee train_f3_1.log

