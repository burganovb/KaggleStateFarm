"""
Created on Tue Jul 12 13:38:03 2016

@author: burganovb
"""

# create LMDB
import numpy as np
import caffe
import lmdb
import os, time, datetime
import math
#from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import random
import cv2
import pandas as pd

import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum

import load_googlenet


def get_train_image_list():
    image_list = []
    path = os.path.join('..','f1','driver_imgs_list.csv')
    print 'Read drivers data'
    f = open(path, 'r')
    line = f.readline()
    
    cnt = 0
    while(1):
        cnt +=1
        #if cnt>100:
        #    break #testing
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        image_list.append(tuple(arr))
    f.close()
    return image_list

def get_test_image_list():
    image_list = []
    path = os.path.join('..','f1','sample_submission.csv')
    print 'Read test image list'
    f = open(path, 'r')
    line = f.readline()
    
    while(1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        image_list.append(arr[0])
    f.close()
    return image_list

def get_im_cv2(path, img_rows=224, img_cols=224, randRGB=False, adjustmean=True, randRotation=False, randCrop=False, randResize=False, driver_clss_masks=None, color_type=1):
    # Load as grayscale
    if color_type == 0:
        #img = np.expand_dims(cv2.imread(path, 0),2)
        img = cv2.imread(path, 0)
    else:
        img = cv2.imread(path)

    if randRGB:
        newclrind = np.random.permutation([0,1,2])
        img = img[:,:,newclrind]
    # modify image mean to get same brightness for all images    
    if adjustmean:
        mu = img.mean(0).mean(0)-np.array([104,116,122]) #make same mu as in ilsvrc_2012_mean.npy
        image = img - np.expand_dims(np.expand_dims(mu,0),0)
        img = np.clip(image,0,255).astype(np.uint8)

    # Apply mask to focus attention on relevant image part    
    if driver_clss_masks is not None:
        driver,clss,masks=driver_clss_masks
        mask = np.expand_dims(masks[tuple((driver,clss))],axis=2)
        #print mask.shape
        img1=cv2.resize(img,(64,48), interpolation = cv2.INTER_AREA)
        img1=cv2.resize(img1,(640,480), interpolation = cv2.INTER_LINEAR)
        img = img*mask+img1*(1-mask)

    if randRotation:
        rotate = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotate, 1)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    if randCrop:
        img = img[random.randint(0,20):random.randint(460,480),random.randint(0,20):random.randint(620,640),:]

    if randResize:
        img = cv2.resize(img, (img_cols, img_rows), interpolation = cv2.INTER_AREA)
        
    resized = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR) #for googlenet needs to be 224x224
    
    #cv2.imwrite('imgout', resized)
    resized = resized.transpose((2,0,1))
    return resized

def generate_crop_masks():
    box = {}
    box['c0']=np.array([400,640,40,330])
    box['c1']=np.array([350,570,90,330])
    box['c2']=np.array([130,400,20,380])
    box['c3']=np.array([350,540,70,260])
    box['c4']=np.array([200,440,20,240])
    box['c5']=np.array([300,640,240,480])
    box['c6']=np.array([170,480,20,360])
    box['c7']=np.array([40,440,50,480])
    box['c8']=np.array([150,480,10,360])
    box['c9']=np.array([120,380,10,300])
    
    masks=dict()
    edge = 25

    drvlist = ['p002','p012','p014','p015','p016','p021','p022','p024','p026','p035','p039','p041','p042','p045','p047','p049','p050','p051','p052','p056','p061','p064','p066','p072','p075','p081']
    for driver in drvlist:
        for clss in ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']:
            Dbox = np.array([0,0,0,0])
            if driver in ('p021','p022','p024','p026'):
                Dbox = np.array([-20,-20,40,40])
            elif driver in ('p035','p039','p041','p042','p045','p047','p049','p050','p051','p052','p056'):
                Dbox = np.array([-60,-60,60,60])
            elif driver in ('p061','p064','p066','p072','p075','p081'):
                Dbox = np.array([-85,-65,0,45])
    
            xy = box[clss]+Dbox
            xy = np.clip(xy,[0,0,0,0],[640,640,480,480])
            
            mask = np.zeros((480,640,1))
            mask[xy[2]:xy[3],xy[0]:xy[1],:]=1.
            kernel = np.ones((2*edge+1,2*edge+1),np.float32)/(2*edge+1)**2
            mask = cv2.filter2D(mask,-1,kernel)
            masks[tuple((driver,clss))]=mask
    return masks

def write_train_val_lmdb_2(val_drivers, lmdbrepeats=1, img_rows=224, img_cols=224, color_type=1, randRotation=True):
    all_images = get_train_image_list() #format: list of (driverid, classname, imagefilename)
    masks = generate_crop_masks()    
    
    #random_state = 51
    img_train = []
    img_val = []    
    for driver, class_name, img_name in all_images:
        if driver in val_drivers:
            img_val.append((driver, class_name, img_name))
        else:
            img_train.append((driver, class_name, img_name))
    print 'all data size', len(all_images)
    print 'train set size', len(img_train)
    print 'val set size', len(img_val)
    
    print 'Writing train images to lmdb ...'
    train_db_name = 'train_lmdb'
    env = lmdb.Environment(train_db_name, map_size=1e12)
    txn = env.begin(write=True,buffers=True)
    idx=0
    
    start_time = time.time()    
    for lmdbcycle in xrange(lmdbrepeats):
        img_imdices = np.random.permutation(len(img_train))
        for kk in img_imdices:
            driver, class_name, img_name = img_train[kk]
            path = os.path.join('..','f1', 'train', class_name, img_name)
            if random.uniform(0,1)<0.4:
                driver_clss_masks = tuple((driver,class_name,masks))
            else:
                driver_clss_masks = None
            randResize = (random.uniform(0,1)<0.35)
            X = get_im_cv2(path, img_rows, img_cols, randRGB=True, adjustmean=True, randRotation=True, randCrop=True, randResize=randResize, driver_clss_masks=driver_clss_masks, color_type=color_type)
            y = int(class_name[1])
        
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())  
            idx+=1
            if idx%1000==0:
                print idx

    txn.commit()
    env.close()
    print 'Writing to ', train_db_name, ' done in {} seconds'.format(round(time.time() - start_time, 2))
    
    
    print 'Writing val images to lmdb ...'
    val_db_name = 'val_lmdb'
    env = lmdb.Environment(val_db_name, map_size=1e12)
    txn = env.begin(write=True,buffers=True)
    idx=0
    
    start_time = time.time()    
    img_imdices = np.random.permutation(len(img_val))
    for kk in img_imdices:
    #for driver, class_name, img_name in img_val:
        driver, class_name, img_name = img_val[kk]
        path = os.path.join('..','f1', 'train', class_name, img_name)
        X = get_im_cv2(path, img_rows, img_cols, randRGB=False, adjustmean=True, randRotation=False, randCrop=False, randResize=False, driver_clss_masks=None, color_type=color_type)
        y = int(class_name[1])
        
        datum = array_to_datum(X, y)
        str_id = '{:08}'.format(idx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())   
        idx+=1

    txn.commit()
    env.close()
    print 'Writing to ', val_db_name, ' done in {} seconds'.format(round(time.time() - start_time, 2))
    
def write_test_lmdb(img_rows=224, img_cols=224):
    all_images = get_test_image_list() #format: list of imagefilename strings
    
    print 'Writing test images to lmdb ...'
    test_db_name = '../test_lmdb'
    env = lmdb.Environment(test_db_name, map_size=1e12)
    #with env.begin(write=True) as txn:
    txn = env.begin(write=True,buffers=True)
    idx=0
    
    start_time = time.time()    
    for img_name in all_images:
        path = os.path.join('..','f1', 'test', img_name)
        #X = get_im_cv2(path, img_rows, img_cols, adjustmean=True, randRotation=False, randScale=False, driver_clss_masks=None, color_type=1)        
        #X = get_im_cv2(path, img_rows, img_cols)
        X = get_im_cv2(path, img_rows, img_cols)

        datum = array_to_datum(X)
        str_id = '{:08}'.format(idx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())  
        idx+=1
        if idx%1000==0:
            print idx

    txn.commit()
    env.close()
    print 'Writing to ', test_db_name, ' done in {} seconds'.format(round(time.time() - start_time, 2))

def read_images_from_lmdb(db_name, visualize=True):
    env = lmdb.open(db_name, readonly=True)
    X = []
    y = []
    idxs = []
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            #print(key, value)
            print type(value), len(value)
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            print type(datum)
            #print datum
            X_1 = datum_to_array(datum)
            X.append(np.array(X_1))
            y.append(datum.label)
            #idxs.append(idx)
            print X_1.shape, datum.label

    if visualize:
        print "Visualizing a few images..."
        '''        
        plt.subplot(1,1,1)
        plt.imshow(X[0].transpose((1,2,0)))
        plt.title(y[0])
        plt.show()
        '''
        for i in range(9):
            img = X[i].transpose((1,2,0))
            plt.subplot(3,3,i+1)
            plt.imshow(img)
            plt.title(y[i])
            plt.axis('off')
        plt.show()
        
    print " ".join(["Reading from", db_name, "done!"])
    return X, y, idxs
    
def generate_train_val_predictions_from_lmdb(db_name, netweights):
    
    print "Creating Net instance"
    caffe_root = os.path.expanduser('~/caffe/')
    net, transformer = load_googlenet.load_googlenet(caffe_root,netweights, gpu=False)
    net.blobs['data'].reshape(1, 3, 224, 224)

    env = lmdb.open(db_name, readonly=True)
    predictions = []
    y = []
    idx = 0
    sum_loss = 0.
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            X_1 = datum_to_array(datum)
            net.blobs['data'].data[0, ...] = X_1
            probsdict = net.forward()
            probs = probsdict['prob'][0]
            probs = postprocess_probs(probs)
            predictions.append(probs)
            y_1 = datum.label
            y.append(y_1)
            idx+=1
            sum_loss -= math.log(probs[y_1])
            if idx%500==0:
                print "Read", idx, ". Loss so far", sum_loss/idx
    
    print 'idx ', idx
    print " ".join(["Reading from", db_name, "done!"])
    print "Loss: ", sum_loss/idx
    return predictions, y

def postprocess_probs(probs):
    probs[probs<0.01] = 0.01
    #probs[probs>0.97] = 0.97
    probsum = np.sum(probs)
    return probs/probsum
    
def generate_test_predictions(netweights, model_def):
    test_image_filenames = get_test_image_list() #format: list of imagefilename strings

    print "Creating Net instance"
    caffe_root = os.path.expanduser('~/loc/caffe/')
    net, transformer = load_googlenet.load_googlenet(caffe_root, netweights, model_def, gpu=True)
    net.blobs['data'].reshape(1, 3, 224, 224)
    
    predictions = []
    idx = 0
    for img_name in test_image_filenames:
        path = os.path.join('..','f1', 'test', img_name)
        #X_1 = get_im_cv2(path, img_rows, img_cols)
        #net.blobs['data'].data[0, ...] = X_1
        #probsdict = net.forward()
        #probs = probsdict['prob'][0]
        net.blobs['data'].data[0, ...] = transformer.preprocess('data', caffe.io.load_image(path))
        net.forward()
        probs = net.blobs['prob'].data[0, ...]
        probs = postprocess_probs(probs)
        predictions.append(probs)
        idx += 1
        if idx%1000==0:
            print "Read", idx, "so far."
	#if idx>200:
	#    break
    return test_image_filenames, predictions

def generate_test_preds_lmdb(netweights, model_def):
    caffe.set_mode_gpu()
    print "Creating Net instance"
    net = caffe.Net(model_def, netweights, caffe.TEST)
    predictions = np.zeros((1,10))
    idx = 0
    while(1):
        net.forward()
        probs = net.blobs['prob'].data
        probs[probs<0.01] = 0.01
        predictions = np.vstack((predictions,probs))
        idx += 50
        if idx%1000==0:
            print "Read", idx, "so far."
        if idx>79726:
            break
    return predictions

def generate_test_3preds_lmdb(netweights, model_def):
    #generates predictions from all three output layers
    caffe.set_mode_gpu()
    print "Creating Net instance"
    net = caffe.Net(model_def, netweights, caffe.TEST)
    predictions_loss3 = np.zeros((0,10))
    predictions_loss1 = np.zeros((0,10))
    predictions_loss2 = np.zeros((0,10))
    #ip1 = np.zeros((0,10))
    #ip2 = np.zeros((0,10))
    #ip3 = np.zeros((0,10))
    idx = 0
    while(1):
        net.forward()
        probs_loss3 = net.blobs['prob'].data
        probs_loss3[probs_loss3<0.008] = 0.008
        predictions_loss3 = np.vstack((predictions_loss3,probs_loss3))
        probs_loss1 = net.blobs['prob_loss1'].data
        probs_loss1[probs_loss1<0.008] = 0.008
        predictions_loss1 = np.vstack((predictions_loss1,probs_loss1))
        probs_loss2 = net.blobs['prob_loss2'].data
        probs_loss2[probs_loss2<0.008] = 0.008
        predictions_loss2 = np.vstack((predictions_loss2,probs_loss2))
        
        #ip1 = np.vstack((ip1, net.blobs['loss1/classifier'].data[:,0:10]))
        #ip2 = np.vstack((ip2, net.blobs['loss2/classifier'].data[:,0:10]))
        #ip3 = np.vstack((ip3, net.blobs['loss3/classifier/new'].data[:,0:10]))
        
        idx += 50
        if idx%1000==0:
            print "Read", idx, "so far."
        if idx>79726:
            break
    #ip = ip1/np.linalg.norm(ip1)+ip2/np.linalg.norm(ip2)+ip3/np.linalg.norm(ip3)
    #exps = np.exp(ip)
    #preds = exps/np.expand_dims(np.sum(exps,axis=1), axis=1)
    
    return predictions_loss1, predictions_loss2, predictions_loss3#, preds

def generate_trainval_predictions(netweights):
    all_images = get_train_image_list() #format: list of (driverid, classname, imagefilename)
    
    random_state = 51
    img_train, img_val = train_test_split(all_images, test_size=0.15, random_state=random_state)
    
    print "Creating Net instance"
    caffe_root = os.path.expanduser('~/caffe/')
    net, transformer = load_googlenet.load_googlenet(caffe_root, netweights, gpu=False)
    net.blobs['data'].reshape(1, 3, 224, 224)

    predictions = []
    image_names = []
    y = []    
    idx=0
    sum_loss = 0
    start_time = time.time()    
    for driver, class_name, img_name in img_train:
        path = os.path.join('imgs', 'train', class_name, img_name)
        #X = get_im_cv2(path)
        y_1 = int(class_name[1])
        #net.blobs['data'].data[0, ...] = X   
        net.blobs['data'].data[0, ...] = transformer.preprocess('data', caffe.io.load_image(path))
        net.forward()
        probs = net.blobs['prob'].data[0, ...]
        print probs.shape
        #probs = probsdict['prob'][0]
        #probs = postprocess_probs(probs)
        predictions.append(probs)
        y.append(y_1)
        image_names.append(img_name)
        print img_name, probs, y_1
        idx += 1
        sum_loss -= math.log(probs[y_1])
        if idx>40:
            break
        #if idx%500==0:
            
    print "Read ", idx, ". Loss so far ", sum_loss/idx
    create_submission(np.array(predictions), image_names, 'train', y)


def create_submission(predictions, test_id, info, labels=None):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    #result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    if labels != None:
        result1.loc[:, 'class'] = pd.Series(labels[0:201], index=result1.index)
    now = datetime.datetime.now()
    if not os.path.isdir('subm'):
        os.mkdir('subm')
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
    result1.to_csv(sub_file, index=False)


#netweights = 'bvlc_googlenet_quick_new_iter_1000000.caffemodel'
#netweights = 'bvlc_googlenet_quick_new_iter_520000.caffemodel'

#generate_trainval_predictions(netweights)
#netweights = '../f1/bvlc_googlenet_quick_new_iter_2400000.caffemodel'
#netweights = 'bvlc_googlenet_quick_new_iter_200000.caffemodel'
#model_def = 'deploy_1.prototxt'
#test_image_filenames, predictions = generate_test_predictions(netweights, model_def)
#create_submission(np.array(predictions), test_image_filenames, 'test')
'''
netwlist = []
netwlist.append('googlenet_2k_step2_iter_6800.caffemodel')
netwlist.append('googlenet_2k_step3_iter_6900.caffemodel')
#netweights = 'googlenet_2k_step1_iter_10000.caffemodel'
netwlist.append('googlenet_2k_step4_iter_7100.caffemodel')
netwlist.append('googlenet_2k_step5_iter_7200.caffemodel')
model_def = 'test_only.prototxt'
for netweights in netwlist:
    pred1,pred2,pred3 = generate_test_3preds_lmdb(netweights, model_def)
    predavg=pred1*0.3+pred2*0.3+0.4*pred3
    create_submission(predavg, None, 'avg1k_stepX')
    predavg=pred1*0.1+pred2*0.45+0.45*pred3
    create_submission(predavg, None, 'avg2k_stepX')
#create_submission(pred1, None, 'loss')
#create_submission(pred2, None, 'loss2')
#create_submission(pred3, None, 'loss3')

'''
#model_def = 'deploy_1.prototxt'
#test_image_filenames, predictions = generate_test_predictions(netweights, model_def)
#create_submission(np.array(predictions), test_image_filenames, 'test')

#val_drivers = ['p014','p022','p041','p075']
#write_train_val_lmdb_2(val_drivers, lmdbrepeats=3, img_rows=100, img_cols=100)
#val_drivers = ['p012','p021','p047','p081']
#val_drivers = ['p015','p026','p051','p064']
#val_drivers = ['p016','p042','p039','p066']
#write_train_val_lmdb_2(val_drivers, lmdbrepeats=3, img_rows=120, img_cols=120)
#write_train_val_lmdb()
write_test_lmdb(img_rows=100,img_cols=100)
#read_images_from_lmdb('train_lmdb', visualize=True)

'''
def load_test(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        # img = get_im_cv2(fl, img_rows, img_cols, color_type)
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


def read_and_normalize_test_data(img_rows, img_cols, color_type=1):
    cache_path = os.path.join('cache', 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color_type) + '_rotated.dat')
    if not os.path.isfile(cache_path) or use_cache == 0:
        test_data, test_id = load_test(img_rows, img_cols, color_type)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if color_type == 1:
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    else:
        test_data = test_data.transpose((0, 3, 1, 2))
    # test_data = test_data.swapaxes(3, 1)
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


    test_data, test_id = read_and_normalize_test_data(img_rows, img_cols, color_type_global)

    for train_drivers, test_drivers in kf:
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=1)
        yfull_test.append(test_prediction)
    test_res = merge_several_folds_mean(yfull_test, nfolds)
    # test_res = merge_several_folds_geom(yfull_test, nfolds)
    create_submission(test_res, test_id, info_string)

'''






'''    
    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, X_train_id, driver_id, unique_drivers

def write_test_lmdb(img_rows, img_cols, color_type=1):
    print('Read test images')
    path = os.path.join('..', 'input', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    start_time = time.time()
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        # img = get_im_cv2(fl, img_rows, img_cols, color_type)
        img = get_im_cv2_mod(fl, img_rows, img_cols, color_type)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    print('Read test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_test, X_test_id


in_db = lmdb.open('image-lmdb', map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, in_ in enumerate(inputs):
        # load image:
        # - as np.uint8 {0, ..., 255}
        # - in BGR (switch from RGB)
        # - in Channel x Height x Width order (switch from H x W x C)
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        im = im[:,:,::-1]
        im = im.transpose((2,0,1))
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
in_db.close()


import numpy as np
import lmdb
import caffe

N = 1000

# Let's pretend this is interesting data
X = np.zeros((N, 3, 32, 32), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
map_size = X.nbytes * 10

env = lmdb.open('mylmdb', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        
################################
        
import numpy as np
import lmdb
import caffe

env = lmdb.open('mylmdb', readonly=True)
with env.begin() as txn:
    raw_datum = txn.get(b'00000000')

datum = caffe.proto.caffe_pb2.Datum()
datum.ParseFromString(raw_datum)

flat_x = np.fromstring(datum.data, dtype=np.uint8)
x = flat_x.reshape(datum.channels, datum.height, datum.width)
y = datum.label

#Iterating <key, value> pairs is also easy:

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        print(key, value)    



def write_images_to_lmdb(img_dir, db_name):
    for root, dirs, files in os.walk(img_dir, topdown = False):
        if root != img_dir:
            continue
        map_size = 64*64*3*2*len(files)
        env = lmdb.Environment(db_name, map_size=map_size)
        txn = env.begin(write=True,buffers=True)
        for idx, name in enumerate(files):
            X = mp.imread(os.path.join(root, name))
            y = 1
            datum = array_to_datum(X,y)
            str_id = '{:08}'.format(idx)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())   
    txn.commit()
    env.close()
    print " ".join(["Writing to", db_name, "done!"])
 '''   
