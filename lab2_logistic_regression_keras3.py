##########################################################################################################################################################
# MNIST Deep Neural Network exploration script
# (C) Copyright 2023, Boris Emchenko
#
# Dataset: MNIST
# Try to create range of models with different hyperparameters with ability to evaluate it accuracy
#
# Possible hyperparameters variations:
#
# - hidden layers (from 1 to 5) with range of possible neurons
# - different activation functions
# - range of epochs
# - range of batch size
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
import time

# Define model
def DefineFFNNModel(layer1_neurons, layer2_neurons=0, layer3_neurons=0, layer4_neurons=0, layer5_neurons=0, layer_activation="sigmoid"):

    print()
    print(CHEAD + "*"*5 + "Let's define model:" + "*"*5  + CEND)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    if (layer1_neurons > 0): model.add(tf.keras.layers.Dense(layer1_neurons, activation=layer_activation, name = 'Hidden1'))
    if (layer2_neurons > 0): model.add(tf.keras.layers.Dense(layer2_neurons, activation=layer_activation, name = 'Hidden2'))
    if (layer3_neurons > 0): model.add(tf.keras.layers.Dense(layer3_neurons, activation=layer_activation, name = 'Hidden3'))
    if (layer4_neurons > 0): model.add(tf.keras.layers.Dense(layer4_neurons, activation=layer_activation, name = 'Hidden4'))
    if (layer5_neurons > 0): model.add(tf.keras.layers.Dense(layer5_neurons, activation=layer_activation, name = 'Hidden5'))
    model.add(tf.keras.layers.Dense(10, activation='softmax', name = 'Output'))

    model.summary()
    return model

def DisplayNeurons(Weights):

    n_to_show = Weights.shape[1]
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for img in enumerate(Weights[:,]):
        print ("* ")


#################################################################################
#   Main
#################################################################################
if __name__ == "__main__":
    # Check command line arguments
    ParseCommandLine(Config)
    print (Config.print())

    # Load dataset
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = LoadDataset(random_state = Config.paramDatasetShuffle)

    #ViewSample(x_train, y_train, [0,1])
    #plt.show()

    # Normalize
    x_train, x_val, x_test = x_train / 255., x_val / 255., x_test / 255.


    # TRAIN AGAIN (or load)?
    if Config.modTrainMode:

        # RUN 
        layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons = 100, 0, 0, 0, 0

        l1_range = [100]
        l2_range = [0]
        l3_range = [0]
        l4_range = [0]
        l5_range = [0]
        epochs_range = [Config.optTargetEpochs]
        activations_range = ["relu"]
        batchsize_range = [Config.optBatchSize]

        l1_range = [100]
        l2_range = [30]

        cnt = 0
        total = len(l1_range)*len(l2_range)*len(l3_range)*len(l4_range)*len(l5_range) * len(epochs_range)*len(activations_range)*len(batchsize_range)
        for layer1_neurons in l1_range:
            for layer2_neurons in l2_range:
                for layer3_neurons in l3_range:
                    for layer4_neurons in l4_range:
                        for layer5_neurons in l5_range:
                            for cur_Epochs in epochs_range:
                                for activ in activations_range:
                                    for cur_BatchSize in batchsize_range:

                                        start_time_train = datetime.now()
                                        cnt = cnt + 1
                                        runtime = (datetime.now() - start_time).total_seconds()
                                        time_left = runtime / cnt * total
                                        print()
                                        print(CHEAD + "*"*50 + CEND)
                                        print(CHEAD + " Iteration %s of %s (%.1f%%)" % (cnt, total, cnt/total*100.0) + CEND)
                                        print(CHEAD + " Runtime: %s, ETA: %s " % (time.strftime("%H:%M:%S", time.gmtime(runtime)), time.strftime("%H:%M:%S", time.gmtime(time_left)))+ CEND)
                                        print(CHEAD + "*"*50 + CEND)

                                        dt = start_time.strftime("%Y%m%d_%H%M%S")
                                        model_folder = "model_FF_" + dt
                                        modelname = "model-seed_%s_e_%s-bs_%s-act_%s-l1_%s-l2_%s-l3_%s-l4_%s-l5_%s" % (Config.paramDatasetShuffle, cur_Epochs, cur_BatchSize, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons)
                                        print(CHEAD + "%s" % (modelname) + CEND)


                                        # Define model
                                        model = DefineFFNNModel(layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, layer_activation =activ)

                                        #Train model
                                        hist = Train_Model(model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, Num_epochs=cur_Epochs, ext_batch_size=cur_BatchSize, auto_stop = Config.optTrainAutoStop, model_file_path=model_folder)
                                        accuracy = hist.history['accuracy']
                                        loss = hist.history['loss']

                                        train_duration = datetime.now() - start_time_train
                                        print(CDARKGREY,'Train duration: {}'.format( train_duration ),CEND)

                                        # Test model
                                        if Config.optSkipTest == False:
                                            test_loss, test_accuracy = Test_Model(model, x_test, y_test)
                                        else:
                                            test_loss = test_accuracy = 0

                                        # SaveModel
                                        # obsolete, now we autosave the model during training
                                        #model.save(model_folder + "\\" + modelname, overwrite=True)
                                        SaveStat(cnt, modelname, model.count_params(), cur_Epochs, cur_BatchSize, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, train_duration, test_loss, test_accuracy, accuracy, loss, model_folder)

                                        saveAccLossGraphs(hist, test_loss, test_accuracy, modelname, model_folder)

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
        VisualizeTest_ConfusionMatrix(predictions, y_test, modelname, model_folder)

    if (Config.optDispalyGraphs or Config.optDisplayLayers):
        plt.show()

    duration = datetime.now() - start_time
    print('Total duration: {}'.format(duration))

