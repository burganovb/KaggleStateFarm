# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 14:58:05 2016

@author: burganovb
"""
import numpy as np


def patch_googlenet(model_def):
    
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(model_def).read(), model)
    model.force_backward = True
    
    
    model.layer[0].input_param.shape[0].dim[0] = 1  # set batchsize to 1
    
    
    open('patched_googlenet.prototxt', 'w').write(str(model))
