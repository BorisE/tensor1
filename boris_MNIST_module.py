##########################################################################################################################################################
# MNIST Deep Neural Network module
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

from datetime import datetime
start_time = datetime.now()


import numpy as np
from matplotlib import pyplot as plt

import csv

import sys

import os
os.system("")
CHEAD = '\033[92m'
CWRONG = '\033[2;31;43m'
CRED = '\033[91m'
CYEL = '\033[33m'
CBLUE = '\033[94m'
CPURPLE = '\033[95m'
CDARKCYAN = '\033[36m'
CLIGHTGREY = '\033[37m'
CDARKGREY = '\033[90m'
CBOLD = '\033[1m'
CEND = '\033[0m'
# https://i.stack.imgur.com/j7e4i.gif
# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
'''
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

sys_details = tf.sysconfig.get_build_info()

print(CHEAD + "*"*50)
print(" We're using TF", tf.__version__)
print(" Keras: ", keras.__version__)
print(" Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(" CUDA: ", sys_details["cuda_version"])
print(" CUDNN: ", sys_details["cudnn_version"])
print("*"*50, CEND)

# Default parameters
default_Epochs = 50
default_BatchSize = 128


class Config:
    modTrainMode = True                 # train model, default
    modLoadModel = False                # don't train, load from disk
    paramModelName = ""                 # model name to load from disk
    optSaveModel = True                 # option: save model to disk after training
    optTargetEpochs = default_Epochs    # param: number of epochs
    optBatchSize = default_BatchSize    # param: batch size
    optTrainAutoStop = True             # param: auto stop training if val accurace stops increasing
    optDispalyGraphs = True             # option: don't dipslay plots after training
    optSkipPredictions = False          # option: don't dipslay plots after training
    optSkipTest = False                 # param: skip model testing
    optDisplayLayers = False            # param: visualize filters/feature maps
    paramDatasetShuffle = None          # param: none - random shuffle, int - shuffle seed (i.e. the same shuffle for the same id)

    @classmethod
    def print(cls):
        s=""
        s=s+"modTrainMode: %s\n" % cls.modTrainMode
        s=s+"optTargetEpochs: %s\n" % cls.optTargetEpochs
        s=s+"optBatchSize: %s\n" % cls.optBatchSize
        s=s+"modLoadModel: %s\n" % cls.modLoadModel
        s=s+"paramModelName: %s\n" % cls.paramModelName
        s=s+"optSaveModel: %s\n" % cls.optSaveModel
        s=s+"optDispalyGraphs: %s\n" % cls.optDispalyGraphs
        s=s+"optTrainAutoStop: %s\n" % cls.optTrainAutoStop
        s=s+"optSkipTest: %s\n" % cls.optSkipTest
        s=s+"optDisplayLayers: %s\n" % cls.optDisplayLayers
        s=s+"paramDatasetShuffle: %s\n" % cls.paramDatasetShuffle
        
        return s

def ParseCommandLine(Config):
    # default parameters values
    global default_Epochs, default_BatchSize
        
    print(CEND)
    print ("Script: ", CHEAD, sys.argv[0], CEND)
    print("Arguments passed: ", end = CHEAD)
    for i in range(1, len(sys.argv)):
        print(sys.argv[i], end = " ")
    print()

    print(CHEAD)

    for i in range(1, len(sys.argv)):
        if "train" in sys.argv[i]:
            Config.modTrainMode = True
            print ("Train mode ON")
        elif "load" in sys.argv[i]:
            Config.modLoadModel = True
            Config.paramModelName = sys.argv[i + 1] if i+1<len(sys.argv) else ""
            print ("Load model mode")
        elif "help" in sys.argv[i]:
            print ("Usage parameters:")
            print (" train \t\t\t train the model and save it")
            print (" --epochs \t\t number of epochs to train (default: %d)" % Config.optTargetEpochs)
            print (" --batchsize \t\t size of batchsize duting train (default: %d)" % Config.optBatchSize)
            print (" --forceallepochs \t\t force train process to run all specified epochs")
            print (" load <d:/python...> \t load the model and run test")
            print (" --skiptest \t\t skip model testing after training or loading")
            print (" --skipsave \t\t don't save trained model")
            print (" --skipgraph \t\t don't dispaly traininig results")
            print (" --skippred \t\t don't make predictions with test vizualization")
            print (" --displayers \t\t visualize layers filters/feature maps")
            print (" --shuffleseed \t\t dataset suffle seed, random for default")
            print (CEND)            
            exit()

        if "--skipsave" in sys.argv[i]:
            Config.optSaveModel = False
            print (" -- Disable saving trained model")
        if "--skiptest" in sys.argv[i]:
            Config.optSkipTest = True
            print (" -- Disable test run for trained/load model")
        if "--skipgraph" in sys.argv[i]:
            Config.optDispalyGraphs = False
            print (" -- Disable displaying training results on graphs")
        if "--skippred" in sys.argv[i]:
            Config.optSkipPredictions = True
            print (" -- Disable predictiction vizualization for test dataset")
        if "--forceallepochs" in sys.argv[i]:
            Config.optTrainAutoStop = False
            print (" -- Train process wouldn't stop automaically on loss stall and will run all specifies epochs")
        if "--epochs" in sys.argv[i]:
            Config.optTargetEpochs = sys.argv[i + 1] if i+1<len(sys.argv) else default_Epochs
            Config.optTargetEpochs = int(Config.optTargetEpochs) if int(Config.optTargetEpochs)>0 else default_Epochs
            print (" -- Set traing epochs to %d" % Config.optTargetEpochs)
        if "--batchsize" in sys.argv[i]:
            Config.optBatchSize = sys.argv[i + 1] if i+1<len(sys.argv) else default_BatchSize
            Config.optBatchSize = int(Config.optBatchSize) if int(type(Config.optBatchSize))>0 else default_BatchSize
            print (" -- Set traing batchsize to ", Config.optBatchSize)
        if "--shuffleseed" in sys.argv[i]:
            Config.paramDatasetShuffle = sys.argv[i + 1] if i+1<len(sys.argv) else None
            Config.paramDatasetShuffle = int(Config.paramDatasetShuffle) if int(Config.paramDatasetShuffle)>0 else None
            print (" -- Set shuffle seed to %d" % Config.paramDatasetShuffle)
        if "--displayers" in sys.argv[i]:
            Config.optDisplayLayers = True
            print (" -- Visualize layers filters/feature maps")


    if Config.modLoadModel:
        Config.modTrainMode = False
        if Config.paramModelName == "":
            Config.paramModelName = "d:/#Dev disk/python/tensorflow-labs/model_20231003_010021/bestmodel.10-0.026"

    print (CEND)

# Load and split dataset
def LoadDataset(random_state = None):

    # Load dataset
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print ("Original dataset: ")
    print(" x_train shape: %s " % (str(x_train.shape)))
    print(" x_test shape: %s" % (str(x_test.shape)))
    print()
    '''
    print("dataset x_train initial shape %s:" % (str(x_train.shape)))
    print (np.bincount(y_train))
    print("y_train [shape %s] first 10 samples:\n" % (str(y_train.shape)),  y_train[:10])
    '''

    # Concatenate alltogether again
    print ("Concatenated dataset: ")
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    print(" x shape %s" % (str(x_all.shape)), end=" | ")
    print("y shape %s" % (str(y_all.shape)), end="")
    print()
    print()

    # Split into train and other (with shuffle)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=random_state, stratify=y_all, shuffle=True)           #70% train, shuffle every run (random_state=None)
    # Split into validation and train (with shuffle)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.1/0.2, random_state=random_state, stratify=y_test, shuffle=True)          #20% validate, 10% test (random_state=None)

    print ("Resplited dataset: ")
    print(" x_train shape %s: " % (str(x_train.shape)), end="\t")
    print (np.bincount(y_train))
    print(" x_val shape %s: " % (str(x_val.shape)), end="\t\t")
    print (np.bincount(y_val))
    print(" x_test shape %s:" % (str(x_test.shape)), end="\t\t")
    print (np.bincount(y_test))
    print()

    print("First 5 test samples (shuffle footprint):")
    print (y_test[:5])
    print()

    return(x_train, y_train), (x_val, y_val), (x_test, y_test)

# Load model from disk
def LoadModel(modelfilename):
    print()
    print(CHEAD + "*"*5 + "Loading pretrained model:" + "*"*5 + CEND)
    print(CHEAD + "*"*3 + modelfilename  + CEND)
    
    saved_model = tf.keras.models.load_model(modelfilename)
    saved_model.summary()


    return saved_model

# View sample data
def ViewSample(dataset_x, dataset_y, sample_id_list):
    #global x_train, y_train, x_val, y_val, x_test, y_test

    #print("Dataset_x [shape %s] sample patch (id=%s):\n" % (str(dataset_x.shape), sample_id), dataset_x[sample_id, 4:21, 4:21])

    '''
    print("A closeup of a sample patch (id=%s): [image]" % (sample_id))
    f1 = plt.figure(1)
    plt.imshow( x_train[sample_id, 4:21, 4:21], cmap="Greys")
    f1.show()
    '''

    n_to_show = len(sample_id_list)
    print("Displaying " + str(n_to_show) + " samples:")

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(sample_id_list):
        img = dataset_x[idx]
        ax = fig.add_subplot(1, n_to_show, i+1)
        ax.axis('off')
        ax.text(0.5, -0.4, 
                'label = ' + str(dataset_y[idx]),
                fontsize=10, 
                ha='center',
                transform=ax.transAxes)

        ax.imshow(img, cmap='binary')
        print(dataset_y[idx], end=' ')

    print()

    #print("Dataset_y [shape %s] first 10 samples:\n" % (str(dataset_y.shape)),  dataset_y[:10])


def Train_Model(model, x_train, y_train, x_val, y_val, Num_epochs=10, ext_batch_size = 32, auto_stop = False, model_file_path = None):

    print()
    print(CHEAD + "*"*5 + "Let's train model:" + "*"*5 + CEND)

    '''
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    '''
    model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # Patience is the number of epochs to wait before stopping the training process if there is no improvement in the model loss
    # EarlyStopping - stop if loss stop decreasing
    # ModelCheckpoint - save models automatically
    callbackslist =  []
    if auto_stop:
        callbackslist.append(tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', verbose=1))
    if model_file_path != None:
        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)
        callbackslist.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_file_path + "/bestmodel.{epoch:02d}-{val_loss:.3f}",
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                verbose = 1,
                save_best_only=True)
        )

    hist = model.fit(x_train, y_train, 
                     validation_data = (x_val, y_val),
                     epochs=Num_epochs, 
                     batch_size=ext_batch_size,
                     callbacks=callbackslist
                     )
    return hist

def Test_Model(model, x_test, y_test, batch_size=32):

    print()
    print(CHEAD + "*"*5 + "Let's test the model:" + "*"*5 + CEND)
  
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

    print(("Test loss: " + CYEL + "%.3f" + CEND + ", test accuracy: " + CYEL + "%.2f%%"+ CEND )% (loss, accuracy*100.0)  )
    
    return loss, accuracy

def Test_makepredictions(model, x_dataset):

    print()
    print(CHEAD + "*"*5 + "Calculate predictions:" + "*"*5 + CEND)

    predictions_raw = model.predict(x_dataset)
    predictions = np.argmax(predictions_raw, axis=1)

    print("Prediction_raw result: " + str(predictions_raw.shape))
    print("Prediction result: " + str(predictions.shape))
    np.set_printoptions(precision=2, suppress=True)
    print(predictions_raw)
    
    return predictions_raw, predictions
def Test_VisualizeText(predictions,  y_dataset, predictions_raw, n_to_show=10):

    erroneous = []
    print()
    print(CHEAD + "*"*5 + "Let's visualize some test results:" + "*"*5 + CEND)
    #print ('\033[2;31;43m' + "*****" + '\033[0m')
    
    pred_sample = np.random.choice(np.arange(len(predictions)), size = n_to_show, replace=False)
    cntwrong = 0
    print("Act:", end=" ")
    for i in pred_sample:
        print(y_dataset[i], end=" ")
    print()
    print("Pre:", end=" ")
    for i in pred_sample:
        if (y_dataset[i] != predictions[i]):
            print (CWRONG, end="")
            cntwrong = cntwrong + 1
            erroneous.append(i)
        print(predictions[i], end=CEND+" ")
    print()
    print("Wrong %s samples out of %s, accuracy = %.1f %%" % (cntwrong, n_to_show, (1-cntwrong/n_to_show)*100))

    return erroneous

def Test_VisualizeGraph(predictions, x_dataset, y_dataset, list_to_show):

    n_to_show = len(list_to_show)
    fig = plt.figure(figsize=(15, 3))
    #fig.title('Test results', fontsize=15)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(list_to_show):
        #print (predictions[i])
        img = x_dataset[idx]
        ax = fig.add_subplot(1, n_to_show, i+1)
        ax.set_title('' + str(predictions[idx]) + ' instead of ' + str(y_dataset[idx]), fontsize = 10)
        ax.axis('off')
        ax.text(0.5, -0.4, 
                'pr = ' + str(predictions[idx]),
                fontsize=10, 
                ha='center',
                transform=ax.transAxes)
        ax.text(0.5, -0.7, 
                'ac = ' + str(y_dataset[idx]),
                fontsize=10, 
                ha='center', 
                transform=ax.transAxes)
        if (predictions[idx] != y_dataset[idx]):
            ax.text(0.5, -1.0, 
                    'WRONG',
                    fontsize=10, 
                    ha='center', 
                    transform=ax.transAxes)

        ax.imshow(img, cmap='binary')

def VisualizeTest_ConfusionMatrix(predictions, y_dataset, modelname="", folder_to_save = None):
    # Convert one-hot encoded labels to integers.
    # Boris: not nesseccary for MNIST
    #y_test_integer_labels = tf.argmax(y_test, axis=1)
    #print (y_test.shape)
    #print (y_test_integer_labels.shape)
    
    # Generate a confusion matrix for the test dataset.
    cm = tf.math.confusion_matrix(labels=y_dataset, predictions=predictions)
    
    # Plot the confusion matrix as a heatmap.
    plt.figure(figsize=[14, 7])

    import seaborn as sn

    sn.heatmap(cm, vmax=10, annot=True, fmt='d', annot_kws={"size": 12})
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    if (folder_to_save != None):
        plt.savefig(folder_to_save + "/confmatrix_" + modelname + '.png')


def SaveStat (cnt, modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration,  test_loss, test_accuracy, accuracy, loss, folder_to_save=""):
    
    row = [datetime.now(), modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_accuracy, test_loss, accuracy, loss]

    dt = start_time.strftime("%Y%m%d_%H%M%S")
    with open(folder_to_save + 'results_' + dt + '.csv', 'a') as f:
        writer = csv.writer(f)
        if (cnt==1):
            writer.writerow(["date","modelname", "parameters count", "epochs", "batch_size", "activation function", "L1 neurons", "L2 neurons", "L3 neurons", "L4 neurons", "L5 neurons", "run_time", "test_accuracy", "test_loss", "accuracy","loss"]) #header
        writer.writerow(row)

def saveAccLossGraphs(hist, test_loss, test_accuracy, modelname="", folder_to_save = None):
    
    if (folder_to_save != None):
        dt = start_time.strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)

    floss = plt.figure()
    floss.set_size_inches((15, 5))
    plt.axis([1, len(hist.history['loss']) + 1, 0, 0.5])
    plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
    plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
    plt.plot([1, len(hist.history['val_loss']) + 1], [test_loss, test_loss], color='blue', linestyle='dashed', linewidth=1)
    plt.title('Loss - Epoch Graphics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Loss', 'Validation Loss', "Test Loss"])
    plt.text(1.1, 0.3, 
                'test = %.3f' % test_loss,
                fontsize=10, 
                ha='left')
    if (folder_to_save != None):
        floss.savefig(folder_to_save + "/loss_" + modelname + '.png')
    #floss.show()

    facc = plt.figure()
    facc.set_size_inches((15, 5))
    plt.axis([1, len(hist.history['accuracy']) + 1, 0.8, 1])
    plt.plot(range(1, len(hist.history['accuracy']) + 1),hist.history['accuracy'])
    plt.plot(range(1, len(hist.history['val_accuracy']) + 1), hist.history['val_accuracy'])
    plt.plot([1, len(hist.history['val_accuracy']) + 1], [test_accuracy, test_accuracy], color='blue', linestyle='dashed', linewidth=1)
    plt.title('Accuracy - Epoch Graphics')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Accuracy', 'Validation Accuracy', "Test Accuracy"])
    plt.text(1.1, 0.9, 
                'test = %.2f%%' % (test_accuracy*100.0),
                fontsize=10, 
                ha='left')
    if (folder_to_save != None):
        facc.savefig(folder_to_save + "/acc_" + modelname + '.png')
    #facc.show()

    #plt.show()
