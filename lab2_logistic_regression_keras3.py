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
##########################################################################################################################################################
import numpy as np
from matplotlib import pyplot as plt

import csv

from datetime import datetime
import time
start_time = datetime.now()

import os
os.system("")
CHEAD = '\033[92m'
CRED = '\033[91m'
CWRONG = '\033[2;31;43m'
CEND = '\033[0m'

import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split

print(CHEAD + "*"*50)
print(" We're using TF", tf.__version__)
print(" Keras: ", keras.__version__)
print(" Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("*"*50)
print(CEND)


# Load and split dataset
def LoadDataset():
    
    # Load dataset
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    '''
    print("dataset x_train initial shape %s:" % (str(x_train.shape)))
    print (np.bincount(y_train))
    print("y_train [shape %s] first 10 samples:\n" % (str(y_train.shape)),  y_train[:10])
    '''

    # Concatenate alltogether again
    x_all = np.concatenate([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    print("x shape %s" % (str(x_all.shape)))
    print("y shape %s" % (str(y_all.shape)))

    # Split into train and other (with shuffle)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=None, stratify=y_all, shuffle=True)           #70% train, shuffle every run (random_state=None)
    # Split into validation and train (with shuffle)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.1/0.2, random_state=None, stratify=y_test, shuffle=True)          #20% validate, 10% test (random_state=None)

    print("x_train shape %s: " % (str(x_train.shape)), end="\t")
    print (np.bincount(y_train))
    print("x_val shape %s:" % (str(x_val.shape)), end="\t")
    print (np.bincount(y_val))
    print("x_test shape %s:" % (str(x_test.shape)), end="\t")
    print (np.bincount(y_test))

    return(x_train, y_train), (x_val, y_val), (x_test, y_test)

# View sample data
def ViewSample(sample_id):
    print("x_train [shape %s] sample patch (id=%s):\n" % (str(x_train.shape), sample_id), x_train[sample_id, 4:21, 4:21])

    '''
    print("A closeup of a sample patch (id=%s): [image]" % (sample_id))
    f1 = plt.figure(1)
    plt.imshow( x_train[sample_id, 4:21, 4:21], cmap="Greys")
    f1.show()
    '''
    print("And the whole sample: [image]")
    f2 = plt.figure(2)
    plt.imshow( x_train[sample_id], cmap="Greys")
    f2.show()

    print("y_train [shape %s] first 10 samples:\n" % (str(y_train.shape)),  y_train[:10])
    print("y_validate [shape %s] first 10 samples:\n" % (str(y_val.shape)),  y_val[:10])


# Define model
def DefineModel(layer1_neurons, layer2_neurons=0, layer3_neurons=0, layer4_neurons=0, layer5_neurons=0, layer_activation="sigmoid"):
    global model

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

def TrainModel(Num_epochs, ext_batch_size = 32):

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

    hist = model.fit(x_train, y_train, 
                     validation_data = (x_val, y_val),
                     epochs=Num_epochs, 
                     batch_size=ext_batch_size)
    return hist

def TestPredict(x_test, y_test):

    print()
    print(CHEAD + "*"*5 + "Let's predict:" + "*"*5 + CEND)
  
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)

    print("Test loss: " + str(loss) + ", test accuracy: " + str(accuracy))
    
    return loss, accuracy

def VisualizeTest_Text(n_to_show=10):
    global pred_sample, erroneous
    global predictions_raw, predictions
    
    print()
    print(CHEAD + "*"*5 + "Let's dispaly some test results:" + "*"*5 + CEND)
    
    predictions_raw = model.predict(x_test)
    predictions = np.argmax(predictions_raw, axis=1)

    np.set_printoptions(precision=2, suppress=True)

    print("Shape: " + str(predictions_raw.shape))
    print(predictions_raw)
    
    print()
    print (CHEAD + "*"*20 + CEND)
    #print ('\033[2;31;43m' + "*****" + '\033[0m')
    
    pred_sample = np.random.choice(np.arange(len(predictions)), size = n_to_show, replace=False)
    cntwrong = 0
    for i in pred_sample:
        print(y_test[i], end=" ")
    print()
    for i in pred_sample:
        if (y_test[i] != predictions[i]):
            print (CWRONG, end="")
            cntwrong = cntwrong + 1
            erroneous.append(i)
        print(predictions[i], end=CEND+" ")
    print()
    print("Wrong %s samples out of %s, accuracy = %.1f %%" % (cntwrong, n_to_show, (1-cntwrong/n_to_show)*100))

def VisualizeTest_Graph(list_to_show):
    global predictions

    n_to_show = len(list_to_show)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(list_to_show):
        #print (predictions[i])
        img = x_test[idx]
        ax = fig.add_subplot(1, n_to_show, i+1)
        ax.axis('off')
        ax.text(0.5, -0.4, 
                'pr = ' + str(predictions[idx]),
                fontsize=10, 
                ha='center',
                transform=ax.transAxes)
        ax.text(0.5, -0.7, 
                'ac = ' + str(y_test[idx]),
                fontsize=10, 
                ha='center', 
                transform=ax.transAxes)
        if (predictions[idx] != y_test[idx]):
            ax.text(0.5, -1.0, 
                    'WRONG',
                    fontsize=10, 
                    ha='center', 
                    transform=ax.transAxes)

        ax.imshow(img, cmap='binary')

    plt.show()

def SaveStat (cnt, modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration,  test_loss, test_accuracy, accuracy, loss):
    
    row = [datetime.now(), modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_accuracy, test_loss, accuracy, loss]

    dt = start_time.strftime("%Y%m%d_%H%M%S")
    with open('results_' + dt + '.csv', 'a') as f:
        writer = csv.writer(f)
        if (cnt==1):
            writer.writerow(["date","modelname", "parameters count", "epochs", "batch_size", "activation function", "L1 neurons", "L2 neurons", "L3 neurons", "L4 neurons", "L5 neurons", "run_time", "test_accuracy", "test_loss", "accuracy","loss"]) #header
        writer.writerow(row)

def saveAccLossGraphs(hist, test_loss, test_accuracy, modelname):
    
    dt = start_time.strftime("%Y%m%d_%H%M%S")
    folder = "figures_" + dt
    if not os.path.exists(folder):
        os.makedirs(folder)

    floss = plt.figure()
    floss.set_size_inches((15, 5))
    plt.axis([1, len(hist.history['loss']) + 1, 0, 1])
    plt.plot(range(1, len(hist.history['loss']) + 1), hist.history['loss'])
    plt.plot(range(1, len(hist.history['val_loss']) + 1), hist.history['val_loss'])
    plt.plot([1, len(hist.history['val_loss']) + 1], [test_loss, test_loss], color='blue', linestyle='dashed', linewidth=1)
    plt.title('Loss - Epoch Graphics')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Loss', 'Validation Loss', "Test Loss"])
    floss.savefig(folder + "/loss_" + modelname + '.png')
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
    facc.savefig(folder + "/acc_" + modelname + '.png')
    #facc.show()

    #plt.show()

#################################################################################
#   Main
#################################################################################

(x_train, y_train), (x_val, y_val), (x_test, y_test) = LoadDataset()

#ViewSample(0)
#plt.show()

# Normalize
x_train, x_val, x_test = x_train / 255., x_val / 255., x_test / 255.

# RUN 
layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons = 100, 0, 0, 0, 0

l1_range = [100]
l2_range = [0]
l3_range = [0]
l4_range = [0]
l5_range = [0]
epochs_range = [5, 15, 30]
activations_range = ["relu"]
batchsize_range = [32, 128, 256]

l1_range = [300, 500, 800]
l2_range = [0, 250, 500, 800]
l3_range = [0, 250, 500, 800]

cnt = 0
total = len(l1_range)*len(l2_range)*len(l3_range)*len(l4_range)*len(l5_range) * len(epochs_range)*len(activations_range)*len(batchsize_range)
for layer1_neurons in l1_range:
    for layer2_neurons in l2_range:
        for layer3_neurons in l3_range:
            for layer4_neurons in l4_range:
                for layer5_neurons in l5_range:
                    for epochs in epochs_range:
                        for activ in activations_range:
                            for bs in batchsize_range:

                                start_time_train = datetime.now()
                                cnt = cnt + 1
                                runtime = (datetime.now() - start_time).total_seconds()
                                time_left = runtime / cnt * total
                                print()
                                print(CHEAD + "*"*50 + CEND)
                                print(CHEAD + " Iteration %s of %s (%.1f%%)" % (cnt, total, cnt/total*100.0) + CEND)
                                print(CHEAD + " Runtime: %s, ETA: %s " % (time.strftime("%H:%M:%S", time.gmtime(runtime)), time.strftime("%H:%M:%S", time.gmtime(time_left)))+ CEND)
                                print(CHEAD + "*"*50 + CEND)

                                modelname = "model-e_%s-bs_%s-act_%s-l1_%s-l2_%s-l3_%s-l4_%s-l5_%s" % (epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons)
                                print(CHEAD + "%s" % (modelname) + CEND)

                                # Define model
                                DefineModel(layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, layer_activation =activ)
                                #Train model
                                hist = TrainModel(Num_epochs=epochs, ext_batch_size=bs)
                                # Test model
                                test_loss, test_accuracy = TestPredict(x_test, y_test)

                                # SaveModel
                                # dt = start_time.strftime("%Y%m%d_%H%M%S")
                                #model.save("models" + dt + "\\" + modelname, overwrite=True)

                                duration = datetime.now() - start_time_train
                                accuracy = hist.history['accuracy']
                                loss = hist.history['loss']
                                SaveStat(cnt, modelname, model.count_params(), epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_loss, test_accuracy, accuracy, loss)
                                saveAccLossGraphs(hist, test_loss, test_accuracy, modelname)

                                print()
                                print(CHEAD + 'Train duration: {}'.format(duration) + CEND)
                                print()



erroneous = []
VisualizeTest_Text(10)
print (erroneous)

VisualizTest_Graph(erroneous)


duration = datetime.now() - start_time
print('Total duration: {}'.format(duration))

