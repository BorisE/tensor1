##########################################################################################################################################################
# CNN test script
# (C) Copyright 2023, Boris Emchenko
#
# Dataset: MNIST
# Based on LAB2 (Logistic Regression module, almost full reuse)
#
# Output:
# - csv file with tested hyperparameters and results
# - graphs of accuracy/loss dynamics for all runs
# - some other visualizations
#
# Tested on tensorflow 2.10, python 3.10
##########################################################################################################################################################
from boris_MNIST_module import *

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
import random

import sys
import math

duration = datetime.now() - start_time
print(CDARKGREY,'Includes part: {}'.format(duration),CEND)

# Default parameters
default_Epochs = 50
default_BatchSize = 128

def AddConvBlock(model, filtersize=3, filternum = 1, convlayers=1, dropout=0.25):

    convID = maxpoolID = dropoutID = 0
    for layer in model.layers:
        # check for convolutional layer
        if "Conv2D_" in layer.name:
            convID = convID + 1
        if "MaxPool_" in layer.name:
            maxpoolID = maxpoolID + 1
        if "Dropout_" in layer.name:
            dropoutID = dropoutID + 1


    for i in range(convlayers):
        convID = convID+1
        model.add(tf.keras.layers.Conv2D(filternum, (filtersize, filtersize), padding='same', activation="relu", name = "Conv2D_" + str(filtersize) + "x" + str(filtersize) + "_" + str(convID)))

    maxpoolID = maxpoolID + 1
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, name = "MaxPool_2x2"+ "_" + str(maxpoolID)))
    
    if (dropout>0):
        dropoutID = dropoutID + 1
        model.add(tf.keras.layers.Dropout(dropout, name = ( "Dropout_%.2f"  + "_" + str(dropoutID) )% dropout))

def DefineCNNModel(model_file_path = None):
    global model
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.InputLayer(input_shape=(28,28,1)))

    AddConvBlock(model, filtersize=3,  filternum =  32, convlayers = 2, dropout=0.25)
    AddConvBlock(model, filtersize=3,  filternum =  64, convlayers = 2, dropout=0.25)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
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

            # normalize the filter values to range 0-1 so we can visualize them
            f_min, f_max = filters.min(), filters.max()
            filters = (filters - f_min) / (f_max - f_min)            
            #print(filters[:,:,:,0])

            channels_to_show = filters.shape[2]
            filters_to_show = filters.shape[-1]
            print ("Num channels %d" % channels_to_show)
            print ("Num filters %d" % filters_to_show)
            idx = 1

            fig = plt.figure(figsize=(20, 20))
            fig.suptitle(layer.name + " weights")
            fig.subplots_adjust(hspace=0.4, wspace=0.4)

            for j in range(channels_to_show):
                for i in range(filters_to_show):

                    flt = filters[:,:,j,i]

                    #print ("chan = %d, filter = %d of %d" % (j, i, filters_to_show))
                    #print (flt.shape)
                    #print (flt)
                    ax = fig.add_subplot(channels_to_show, filters_to_show, idx)
                    ax.set_title( str(j) + "_" + str(i), fontsize = 6.0)
                    ax.axis('off')

                    ax.imshow(flt, cmap='binary')
                    idx = idx + 1
def GetFeatureMaps(model, image, layer_idx=1):
    
    print ("Feature maps for layer " + model.layers[layer_idx].name)
    image_exp = np.expand_dims(image, axis=0) # add first dimension to [batch_size, x, y, c]
    #print(image_exp.shape)
    model2 = tf.keras.models.Model(inputs=model.inputs , outputs=model.layers[layer_idx].output)
    features = model2.predict(image_exp)
    print("features shape: ", features.shape)

    num_maps = features.shape[3]
    num_cols = math.ceil(math.sqrt( num_maps ) * 1.3)
    num_rows = math.ceil(num_maps / num_cols)
    print("total maps count: %d, grid size: %dx%d" % (num_maps, num_rows, num_cols))
    
    # visualize featrure maps
    fig = plt.figure(figsize=(20,15))
    fig.suptitle(model.layers[layer_idx].name + " feature maps")
    
    # add source image as firt image
    ax = fig.add_subplot(num_rows+1, num_cols, 1)
    ax.imshow(image , cmap='gray')

    for i in range(1,num_maps+1):
        plt.subplot(num_rows+1,num_cols,i+num_cols)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(features[0, : , : , i-1] , cmap='gray')



#################################################################################
#   Main
#################################################################################
if __name__ == "__main__":
    # Check command line arguments
    ParseCommandLine(Config)
    print (Config.print())

    # Load dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = LoadDataset(random_state = Config.paramDatasetShuffle)

    # Reshape data for ConvNet format (add fourth dimension with color channels)
    print("Reshaping for ConvNet...")
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val=x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    print(" x_train shape %s" % (str(x_train.shape)))
    print(" x_val shape %s" % (str(x_val.shape)))
    print(" x_test shape %s" % (str(x_test.shape)))
    print()


    #ViewSample(x_train, y_train, [0,1,2,3])
    
    # Normalize
    x_train, x_val, x_test = x_train / 255., x_val / 255., x_test / 255.
    #x_train, x_val = x_train / 255., x_val / 255.


    # TRAIN AGAIN (or load)?
    if Config.modTrainMode:

        # Define a model name/path if you want to save it
        if Config.optSaveModel:
            dt = start_time.strftime("%Y%m%d_%H%M%S")
            modelfolder = "model_CNN_" + dt
            modelname = "CNN_test"
        else:
            modelfolder = None
            modelname = ""

        # Define the model
        model = DefineCNNModel(modelfolder)
       
        #Train model
        hist = Train_Model(model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, Num_epochs=Config.optTargetEpochs, ext_batch_size=Config.optBatchSize, auto_stop = Config.optTrainAutoStop, model_file_path=modelfolder)

        accuracy = hist.history['accuracy']
        loss = hist.history['loss']
        #print ("epochs accuracy: " + str(accuracy))
        #print ("epochs loss: " + str(loss))

        print(CDARKGREY,'Load+train duration: {}'.format( datetime.now() - start_time),CEND)

        # Test model
        if Config.optSkipTest == False:
            test_loss, test_accuracy = Test_Model(model, x_test, y_test)
        else:
            test_loss = test_accuracy = 0
        
        # SaveStat(cnt, modelname, model.count_params(), epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_loss, test_accuracy, accuracy, loss)
        saveAccLossGraphs(hist, test_loss, test_accuracy, modelname, modelfolder)
    else:
        # Load model
        model = LoadModel(Config.paramModelName)

        # Test model
        if Config.optSkipTest == False:
            test_loss, test_accuracy = Test_Model(model, x_test, y_test)
        else:
            test_loss = test_accuracy = 0
        

    if not Config.optSkipPredictions:
        predictions_raw, predictions = Test_makepredictions(model=model, x_dataset=x_test)
        erroneous = Test_VisualizeText(predictions=predictions, predictions_raw=predictions_raw, y_dataset=y_test, n_to_show=100)
        if erroneous:
            Test_VisualizeGraph(predictions=predictions, x_dataset = x_test, y_dataset = y_test, list_to_show = erroneous)
        VisualizeTest_ConfusionMatrix(predictions, y_test, modelname, modelfolder)

    if Config.optDisplayLayers:
        #GetWeights(model, "Conv2D_3x3_1")
        #GetWeights(model, "Conv2D_3x3_2")
        GetFeatureMaps(model, x_test[1,:,:,:], 0)
        #GetFeatureMaps(model, x_test[1,:,:,:], 1)
        GetFeatureMaps(model, x_test[1,:,:,:], 5)


    if (Config.optDispalyGraphs or Config.optDisplayLayers):
        plt.show()

    print('Total duration: {}'.format(datetime.now() - start_time))