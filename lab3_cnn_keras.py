##########################################################################################################################################################
# CNN
##########################################################################################################################################################
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import random

import sys

from lab2_logistic_regression_keras3 import *


def AddConvBlock(filtersize=3, filternum = 1, convlayers=1, dropout=0.25):
    for i in range(convlayers):
        model.add(tf.keras.layers.Conv2D(filternum, (filtersize, filtersize), padding='same', activation="relu", name = "Conv2D_" + str(filtersize) + "x" + str(filtersize)))

    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "MaxPool_2x2"))
    if (dropout>0):
        model.add(tf.keras.layers.Dropout(dropout, name = "Dropout_%.2f" % dropout))

def DefineCNNModel(model_file_path = None):
    global model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(32, (3,3), padding='same', activation="relu", name = "Conv2D_1"))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "MaxPool_1"))
    model.add(tf.keras.layers.Dropout(0.25, name = "Dropout_1"))

    AddConvBlock(5,64)

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5, name = "Dropout_f1"))
    model.add(tf.keras.layers.Dense(10, activation="softmax", name = 'Output'))

    model.summary()

    
    if (model_file_path != None):
        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)        
        tf.keras.utils.plot_model(
            model,
            to_file=model_file_path + "/model.png",
            show_shapes=True,
            show_dtype = False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            show_layer_activations=True,
            dpi=96,
        )
    

    return model

def GetWeights(model, layer_name):

    for layer in model.layers:
        # check for convolutional layer
        if layer_name not in layer.name:
            continue
        
        if 'Conv' in layer.name:
            filters, biases = layer.get_weights()
            print(layer.name, "filters=", filters.shape)
            print(layer.name, "biases=", biases.shape)
            #print(filters[:,:,:,0])

            fig = plt.figure(figsize=(15, 3))
            fig.suptitle("Weights")
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            n_to_show = filters.shape[-1]
            print ("Num filters %d" % n_to_show)
            for i in range(filters.shape[-1]):
                flt = filters[:,:,0,i]
                #print ("filter = %s of %d" % (i, n_to_show))
                #print (flt.shape)
                #print (flt)
                ax = fig.add_subplot(1, n_to_show, i+1)
                ax.set_title('f' + str(i))
                ax.axis('off')

                ax.imshow(flt, cmap='binary')
                i=i+1

    plt.show()

def CheckCommandLine():
    print(CHEAD)
    print("Script name:", sys.argv[0])
    print("Arguments passed:", end = " ")
    for i in range(1, len(sys.argv)):
        print(sys.argv[i], end = " ")
    print()

    TrainMode = False
    if any("train" in s for s in sys.argv[1:]):
        TrainMode = True
        print ("Train mode ON")

    print (CEND)
    return TrainMode

#################################################################################
#   Main
#################################################################################
if __name__ == "__main__":
    # Check command line arguments
    TrainMode = CheckCommandLine()

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = LoadDataset(random_state = 3)

    print("Reshaping for ConvNet...")
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    print(x_train.shape)
    print(x_val.shape)
    print(x_test.shape)

    print("First 5 test samples:")
    print (y_test[:10])

    #ViewSample(x_train, y_train, [0,1,2,3])
    
    # Normalize
    x_train, x_val, x_test = x_train / 255., x_val / 255., x_test / 255.
    #x_train, x_val = x_train / 255., x_val / 255.


    # TRAIN AGAIN (or load)?
    if (TrainMode):

        epochs = 50
        bs = 128

        # Define a model
        dt = start_time.strftime("%Y%m%d_%H%M%S")
        modelfolder = "model_" + dt
        modelname = "CNN_test"

        model = DefineCNNModel(modelfolder)
       
        #Train model
        hist = TrainModel(model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, Num_epochs=epochs, ext_batch_size=bs, use_callbacks = True, model_file_path=modelfolder)

        accuracy = hist.history['accuracy']
        loss = hist.history['loss']
        print ("accuracy: " + str(accuracy) + " loss: " + str(loss))

        duration = datetime.now() - start_time
        print('Load/train duration: {}'.format(duration))

        # Test model
        test_loss, test_accuracy = TestPredict(model, x_test, y_test)
        
        # SaveStat(cnt, modelname, model.count_params(), epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_loss, test_accuracy, accuracy, loss)
        saveAccLossGraphs(hist, test_loss, test_accuracy, modelname, modelfolder)
    else:
        #model = LoadModel("d:/#Dev disk/python/tensorflow-labs/model_20231001_204005/bestmodel.06-0.032") #3x3 filters
        model = LoadModel("d:/#Dev disk/python/tensorflow-labs/model_20231003_002334/bestmodel.05-0.030") #5x5 without dropout
        model = LoadModel("d:/#Dev disk/python/tensorflow-labs/model_20231003_010021/bestmodel.10-0.026") #5x5 with dropout
        # Test model
        test_loss, test_accuracy = TestPredict(model, x_test, y_test)
        
        GetWeights(model, "Conv2D_1")


    predictions, erroneous = VisualizeTest_Text(model=model, x_dataset=x_test, y_dataset=y_test, n_to_show=100)
    if erroneous:
        print (erroneous)
        VisualizeTest_Graph(predictions=predictions, x_dataset = x_test, y_dataset = y_test, list_to_show = erroneous)

    VisualizeTest_ConfusionMatrix(predictions, y_test)

    duration = datetime.now() - start_time
    print('Total duration: {}'.format(duration))