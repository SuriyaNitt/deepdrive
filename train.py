import networks.tflearn.deepdrivenet as ddn #threeD_conv_net
import ml_utils.calc_optical_flow as cop #calc_opticalflow
import ml_utils.handle_data as hd
import os
import numpy as np
import tflearn
import tensorflow as tf
debug = 0
calc_flow = 0
load_model_on_start = 0
depth = 16
batchSize = 16
modelBatchSize = 2
rows = 224
cols = 224
colorSpace = 3

tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.8)
h5FilePath = '/home/suriya/Documents/hard_disk/DeepDrive'
h5FileName = 'train_0001.h5'

h5File = os.path.join(h5FilePath, h5FileName)

video = hd.read_h5_key_value(h5File, 'images')
video  = hd.swap_axes(video, 1, 3)
targets = hd.read_h5_key_value(h5File, 'targets')
targetSpeed = targets[:, 2]
targetSpeed = targetSpeed.reshape((targetSpeed.shape[0],1))
targetSteering = targets[:, 4]
targetSteering = targetSteering.reshape((targetSteering.shape[0], 1))
myTargets = np.append(targetSpeed, targetSteering, axis=1)

if debug:
    print video.shape
    print targets.shape
    print targets[0]
    print myTargets.shape
if calc_flow:
    cop.calc_opticalflow(video, video.shape[0])

myNet = ddn.threeD_conv_net_single_stream(depth, rows, cols, modelBatchSize)
model = tflearn.DNN(myNet, checkpoint_path='./model_resnet',
                    max_checkpoints=10, tensorboard_verbose=3, tensorboard_dir='./tflearn_logs')
if load_model_on_start:
    model.load('./model_resnet/model1')

testX = np.ndarray((0, cols, rows, colorSpace), dtype='float32')
if debug:
    print testX.shape
for i in range(batchSize):
    testX = np.append(testX, video[i:i+depth, :cols, :rows, :], axis=0)
testX = testX.reshape((batchSize, depth, cols, rows, colorSpace))
if debug:
    print testX.shape
testY = myTargets[depth-1:depth-1+batchSize, :]
testY = np.array(testY)
testY = testY.reshape((testY.shape[0], 2))

train_count = 1
if debug:
    train_count = 1
while train_count < ((targetSpeed.shape[0]-depth+1) / batchSize): # 970 / 32
    trainX = np.ndarray((0, cols, rows, colorSpace), dtype='float32')
    for i in range(batchSize):
        trainX = np.append(trainX, video[i+(train_count)*batchSize:i+(train_count)*batchSize+depth, :cols, :rows, :], axis=0)
    trainX = trainX.reshape(batchSize, depth, rows, cols, colorSpace)
    trainY = myTargets[depth-1+(train_count)*batchSize:depth-1+(train_count+1)*batchSize, :]
    trainY = np.array(trainY)
    trainY = trainY.reshape((testY.shape[0], 2))
    if debug:
        print trainX.shape
        print trainY.shape
        print testX.shape
        print testY.shape
    # Training the neural net
    with tf.device('/gpu:0'):
        model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=modelBatchSize, run_id='resnet')
    model.save('./model_resnet/model1')
    train_count += 1
