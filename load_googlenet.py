import os, sys,  math, random, subprocess
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, Image, display, HTML
from google.protobuf import text_format
import PIL.Image

def load_googlenet(caffe_root, model_weights, model_def, gpu=False):
    """
    Load the VGG model in pycaffe, as well as the transformer which can
    convert images to/from the format required by the first layer ("data").

    :returns: (net, transformer)
    """

    # Note: assumes that the python path is configured
    import caffe

    print 'looking for file: ', model_weights
    if os.path.isfile(model_weights):
        print 'GoogLeNet found.'
    else:
        print 'GoogLeNet.caffemodel not found. Please download.'
        return 0
    if os.path.isfile(model_def):
        print 'GoogLeNet definition file found.'
    else:
        print 'GoogLeNet definition not found. Please download.'
        return 0
    
    # 2. Load net and set up input preprocessing
    if gpu:
        caffe.set_mode_gpu()
    else:
        # * Set Caffe to CPU mode and load the net from disk.
        caffe.set_mode_cpu()

    #model_def = caffe_root + 'models/bvlc_googlenet/deploy.prototxt'
    #model_weights = caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'
    #model_def = 'deploy_1.prototxt'
    #model_weights = 'bvlc_googlenet_quick_new_iter_1014.caffemodel'

    '''
    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(model_def).read(), model)
    model.force_backward = True
    #return model
    model.layer[0].input_param.shape[0].dim[0] = 1  # set batchsize to 1
    open('patched_googlenet.prototxt', 'w').write(str(model))

    print 'here1'
    '''
    #net = caffe.Net('patched_googlenet.prototxt', # defines the structure of the model
    net = caffe.Net(model_def, # defines the structure of the model
                    model_weights,              # contains the trained weights
                    caffe.TEST)              # use test mode (e.g., don't perform dropout)
    #print 'here2'

    # * Set up input preprocessing. (We'll use Caffe's `caffe.io.Transformer` to do this,
    # but this step is independent of other parts of Caffe, so any custom preprocessing code may be used).
    # Our default AlexNet is configured to take images in BGR format. Values are expected to start
    # in the range [0, 255] and then have the mean ImageNet pixel value subtracted from them.
    # In addition, the channel dimension is expected as the first (outermost) dimension.

    # load the mean ImageNet image (as distributed with Caffe) for subtraction
    #mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
    #mu = mu.mean(1).mean(1)  # average pixels over to obtain the mean (BGR) pixel values
    mu = np.array([104,117,123])

    # create transformer for the input called 'data'
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    #transformer.set_mean('data', 117)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    # set the size of the input (we can skip this if we're happy
    #  with the default; we can also change it later, e.g., for different batch sizes)
    net.blobs['data'].reshape(1,         # batch size
                            3,         # 3-channel (BGR) images
                            224, 224)  # image size is 224x224
    net.reshape()


    print 'Loaded net', net

    return net, transformer
