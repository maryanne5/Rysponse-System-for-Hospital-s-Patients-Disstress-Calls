import pandas as pd
import numpy as np
from sklearn import preprocessing
from tensorflow import keras
from keras import layers
from tensorflow.python.keras.models import load_model

# Suppress irrelevant warnings
import os

from Utils.graphs import display_model_graphs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

epochsNum = 250


def gen_Learning_data_LOS():
    # Create dataframe from csv
    data = pd.read_csv('..\\Data\\patientsList.csv')
    # Save the length of stay in a different variable
    labels = data['lengthofstay']
    # Drop columns that we dont need like specific dates, or the id of the patient
    data = data.drop(["firstName", "age", "lastName", "room Number", "bed Number", "eid", "vdate", "priority rate",
                      "discharged", "Mortality", "lengthofstay",
                      "visits amount", "days passed since last visit", "emergency button clickes amount"], axis=1)
    # Add dummy encoding for the object and type variables
    # For example, turn gender column into 2 columns, where a male will be 1 in the first column
    # and a 0 in the second column, and a female will be the inverse
    data = pd.get_dummies(data, columns=['rcount'])
    data = pd.get_dummies(data, columns=['gender'])
    data = pd.get_dummies(data, columns=['facid'])

    # Data standardization is the process of rescaling the attributes so that they have mean as 0 and variance as 1.
    # The ultimate goal to perform standardization is to bring down all the features to a common scale without distorting the differences in the range of the values.
    # We use fit_transform() on training data and transform() on the test data
    hematocrit = data[['hematocrit']].values
    data['hematocrit'] = preprocessing.StandardScaler().fit_transform(hematocrit)

    bloodureanitro = data[['neutrophils']].values
    data['neutrophils'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    sodium = data[['sodium']].values
    data['sodium'] = preprocessing.StandardScaler().fit_transform(sodium)

    glucose = data[['glucose']].values
    data['glucose'] = preprocessing.StandardScaler().fit_transform(glucose)

    bloodureanitro = data[['bloodureanitro']].values
    data['bloodureanitro'] = preprocessing.RobustScaler().fit_transform(bloodureanitro)

    creatinine = data[['creatinine']].values
    data['creatinine'] = preprocessing.StandardScaler().fit_transform(creatinine)

    bmi = data[['bmi']].values
    data['bmi'] = preprocessing.StandardScaler().fit_transform(bmi)

    pulse = data[['pulse']].values
    data['pulse'] = preprocessing.StandardScaler().fit_transform(pulse)

    respiration = data[['respiration']].values
    data['respiration'] = preprocessing.StandardScaler().fit_transform(respiration)
    # Seperate for train and test
    train_X = data.head(n=80000).to_numpy()
    train_Y = labels.head(n=80000).to_numpy()

    return train_X, train_Y


# A method to create the initial model
# it is used once at the first time
# receives the training data
# trains the models
# saves the model for later on usage in the Data folder
def create_LOS_Model(train_X, train_Y):
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    model = keras.Sequential()
    model.add(
        layers.LSTM(32, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                    recurrent_dropout=0.5, input_shape=(1, 34)))
    model.add(
        layers.LSTM(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                    recurrent_dropout=0.5, input_shape=(1, 34)))
    model.add(layers.Dense(units=18))
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"])
    history = model.fit(train_X, train_Y, validation_split=0.5, epochs=epochsNum, batch_size=100, verbose=2)
    metric = ['accuracy', 'val_accuracy']
    loss = ['loss', 'val_loss']
    display_model_graphs(history, epochsNum, metric, "Accuracy", 'LOS')
    display_model_graphs(history, epochsNum, loss, "Cross Entropy", 'LOS')
    model.save("..\\Data\\LOSModel.h5")


# This function is to re-Train the existing Length of Stay model
def model_LOS_learning(train_X, train_Y):
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    # Load the previous model
    model = load_model("..\\Data\\LOSModel.h5", compile=False)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"])
    model.fit(train_X, train_Y, validation_split=0.5, epochs=epochsNum, batch_size=100, verbose=2)
    # Save the model
    model.save("..\\Data\\LOSModel.h5")


def LOS_Learning():
    print("start preprocessing")
    Train_X, Train_Y= gen_Learning_data_LOS()
    print("done preprocessing")

    print("Starting LOS Model Learning\n")
    print("----------------------LOS Model Learning Results----------------------------")
    model_LOS_learning(Train_X, Train_Y)
    print("LOS Model Learning Ended\n")


def main():
    LOS_Learning()


if __name__ == '__main__':
    main()
