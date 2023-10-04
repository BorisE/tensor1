
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
CYEL = '\033[33m'
CWRONG = '\033[2;31;43m'
CEND = '\033[0m'

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
print("*"*50)
print(CEND)


# Load and split dataset
def LoadDataset(random_state = None):

    # Load dataset
    from tensorflow.keras.datasets import mnist
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

    # Split into train and other (with shuffle)
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=random_state, stratify=y_all, shuffle=True)           #70% train, shuffle every run (random_state=None)
    # Split into validation and train (with shuffle)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.1/0.2, random_state=random_state, stratify=y_test, shuffle=True)          #20% validate, 10% test (random_state=None)

    print ("Resplited dataset: ")
    print(" x_train shape %s: " % (str(x_train.shape)), end="\t")
    print (np.bincount(y_train))
    print(" x_val shape %s:" % (str(x_val.shape)), end="\t")
    print (np.bincount(y_val))
    print(" x_test shape %s:" % (str(x_test.shape)), end="\t")
    print (np.bincount(y_test))

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

def TrainModel(model, x_train, y_train, x_val, y_val, Num_epochs, ext_batch_size = 32, use_callbacks = False, model_file_path = ""):

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
    if not os.path.exists(model_file_path):
        os.makedirs(model_file_path)
    
    # EarlyStopping - stop if loss stop decreasing
    # ModelCheckpoint - save models automatically
    if (use_callbacks):
        callbackslist = [
             tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss', verbose=1),
             tf.keras.callbacks.ModelCheckpoint(
                filepath=model_file_path + "/bestmodel.{epoch:02d}-{val_loss:.3f}",
                save_weights_only=False,
                monitor='val_loss',
                mode='min',
                verbose = 1,
                save_best_only=True)
        ]
    else:
        callbackslist = []

    hist = model.fit(x_train, y_train, 
                     validation_data = (x_val, y_val),
                     epochs=Num_epochs, 
                     batch_size=ext_batch_size,
                     callbacks=callbackslist
                     )
    return hist

def TestPredict(model, x_test, y_test, batch_size=32):

    print()
    print(CHEAD + "*"*5 + "Let's test the model:" + "*"*5 + CEND)
  
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)

    print(("Test loss: " + CYEL + "%.3f" + CEND + ", test accuracy: " + CYEL + "%.2f%%"+ CEND )% (loss, accuracy*100.0)  )
    
    return loss, accuracy

def VisualizeTest_Text(model, x_dataset,  y_dataset, n_to_show=10):

    erroneous = []

    print()
    print(CHEAD + "*"*5 + "Let's dispaly some test results:" + "*"*5 + CEND)
    
    print("Calculate prediction for a test set...")
    predictions_raw = model.predict(x_dataset)
    predictions = np.argmax(predictions_raw, axis=1)

    np.set_printoptions(precision=2, suppress=True)

    print("Prediction result: " + str(predictions_raw.shape))
    print(predictions_raw)
    
    print()
    print (CHEAD + "*"*20 + CEND)
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

    return predictions, erroneous

def VisualizeTest_Graph(predictions, x_dataset, y_dataset, list_to_show):

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

    plt.show()

def VisualizeTest_ConfusionMatrix(predictions, y_dataset):
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

    plt.show()

def SaveStat (cnt, modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration,  test_loss, test_accuracy, accuracy, loss, folder_to_save=""):
    
    row = [datetime.now(), modelname, modelparameters, epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_accuracy, test_loss, accuracy, loss]

    dt = start_time.strftime("%Y%m%d_%H%M%S")
    with open(folder_to_save + 'results_' + dt + '.csv', 'a') as f:
        writer = csv.writer(f)
        if (cnt==1):
            writer.writerow(["date","modelname", "parameters count", "epochs", "batch_size", "activation function", "L1 neurons", "L2 neurons", "L3 neurons", "L4 neurons", "L5 neurons", "run_time", "test_accuracy", "test_loss", "accuracy","loss"]) #header
        writer.writerow(row)

def saveAccLossGraphs(hist, test_loss, test_accuracy, modelname, folder_to_save):
    
    dt = start_time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)

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
    facc.savefig(folder_to_save + "/acc_" + modelname + '.png')
    #facc.show()

    #plt.show()

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
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = LoadDataset()

    #ViewSample(x_train, y_train, [0,1])
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
    epochs_range = [10]
    activations_range = ["relu"]
    batchsize_range = [128]

    l1_range = [100]
    l2_range = [30]

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

                                    dt = start_time.strftime("%Y%m%d_%H%M%S")
                                    model_folder = "model_" + dt
                                    modelname = "model-e_%s-bs_%s-act_%s-l1_%s-l2_%s-l3_%s-l4_%s-l5_%s" % (epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons)
                                    print(CHEAD + "%s" % (modelname) + CEND)


                                    # Define model
                                    model = DefineFFNNModel(layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, layer_activation =activ)
                                    #Train model
                                    hist = TrainModel(model, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, Num_epochs=epochs, ext_batch_size=bs, model_file_path=model_folder)
                                    # Test model
                                    test_loss, test_accuracy = TestPredict(model, x_test, y_test)

                                    # SaveModel
                                    #model.save(model_folder + "\\" + modelname, overwrite=True)

                                    duration = datetime.now() - start_time_train
                                    accuracy = hist.history['accuracy']
                                    loss = hist.history['loss']
                                    SaveStat(cnt, modelname, model.count_params(), epochs, bs, activ, layer1_neurons, layer2_neurons, layer3_neurons, layer4_neurons, layer5_neurons, duration, test_loss, test_accuracy, accuracy, loss, model_folder)
                                    saveAccLossGraphs(hist, test_loss, test_accuracy, modelname, model_folder)

                                    print()
                                    print(CHEAD + 'Train duration: {}'.format(duration) + CEND)
                                    print()




    predictions, erroneous = VisualizeTest_Text(model=model, x_dataset=x_test, y_dataset=y_test, n_to_show=10)
    print (erroneous)

    VisualizeTest_Graph(predictions=predictions, x_dataset = x_test, y_dataset = y_test, list_to_show = erroneous)


    duration = datetime.now() - start_time
    print('Total duration: {}'.format(duration))

