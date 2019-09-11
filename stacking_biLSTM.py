import numpy as np
import pandas as pd
import copy
import graphviz
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import accuracy_score
import keras.backend as K
from keras import regularizers
from keras.models import Sequential, load_model, Model
from keras.layers import LSTM, Dense, Bidirectional, CuDNNLSTM
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical, plot_model
from matplotlib import pyplot
from os import makedirs
from numpy import dstack
import os

#os.environ['PATH'] += os.pathsep + 'C:/Users/Graphviz2.38/bin/' #Use this code if there is grpahviz error

N = 40000
lookback = 28

##### Please modify prepare_data function for your own data
def prepare_data():
    data = pd.read_csv('train.csv', index_col=(0, 1))

    features = data.columns

    data[features] = np.log(data[features] + 1)

    #     del data['acc_id']
    #     del data['day']

    X_train = data[:int(N * lookback * 0.8)].values
    X_test = data[-int(N * lookback * 0.2):].values

    label = pd.read_csv('label.csv')

    Y_train = label[:int(N * 0.8)].values
    Y_test = label[-int(N * 0.2):].values

    X_train = X_train.reshape(X_train.shape[0] // lookback, lookback, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0] // lookback, lookback, X_test.shape[1])

    return X_train, X_test, Y_train, Y_test

def fit_model(X_train, Y_train):
    model = Sequential()
    model.add(Bidirectional(CuDNNLSTM(128,
                   return_sequences=True,
                   input_shape=(lookback, X_train.shape[2]),
                   kernel_regularizer=regularizers.l2(0.01))))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.01))))
    model.add(Bidirectional(CuDNNLSTM(32, kernel_regularizer=regularizers.l2(0.01))))
    model.add(Dense(64, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, to_categorical(Y_train[:, 0])[:, 1:], epochs=250, batch_size=256, verbose=2,
                         validation_split=0.125)
    return model

def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'models/model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# create stacked model input dataset as outputs from the ensemble
def stacked_dataset(members, inputX):
    stackX = None
    for model in members:
        # make prediction
        yhat = model.predict(inputX, verbose=0)
        # stack predictions into [rows, members, probabilities]
        if stackX is None:
            stackX = yhat
        else:
            stackX = dstack((stackX, yhat))
    # flatten predictions to [rows, members x probabilities]
    stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
    return stackX

# define stacked model from multiple member input models
def define_stacked_model(members):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # make not trainable
            layer.trainable = False
            # rename to avoid 'unique layer name' issue
            layer.name = 'ensemble_' + str(i + 1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(128, activation='relu')(merge)
    output = Dense(64, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # encode output data
    inputy_enc = to_categorical(inputy[:, 0])[:, 1:]
    # fit model
    model.fit(X, inputy_enc, epochs=2, verbose=2)


# make a prediction with a stacked model
def predict_stacked_model(model, inputX):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]
    # make prediction
    return model.predict(X, verbose=0)

X_train, X_test, Y_train, Y_test = prepare_data()
#
# # create directory for models
makedirs('models')
# fit and save models
n_members = 3
for i in range(n_members):
	# fit model
	model = fit_model(X_train, Y_train)
	# save model
	filename = 'models/model_' + str(i + 1) + '.h5'
	model.save(filename)
	print('>Saved %s' % filename)

print(X_train.shape, X_test.shape)
# load all models
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
# define ensemble model
stacked_model = define_stacked_model(members)
plot_model(stacked_model)
# fit stacked model on test dataset
fit_stacked_model(stacked_model, X_test, Y_test)
# make predictions and evaluate
yhat = predict_stacked_model(stacked_model, X_test)
yhat = np.argmax(yhat, axis=1)
acc = accuracy_score(Y_test, yhat)
print('Stacked Test Accuracy: %.3f' % acc)