import keras.metrics
import pandas as pd
import numpy as np
from keras import optimizers
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.python.keras.layers import SimpleRNN, BatchNormalization, Activation, Dropout
from tensorflow.python.keras.models import load_model

from Utils.graphs import display_model_graphs

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

epochsNum = 250


# returns training and testing data separated
def gen_Learning_data_Mortality():
    # Create dataframe from csv
    data = pd.read_csv('..\\Data\\patientsList.csv')
    # Save the length of stay in a different variable
    labels = data['Mortality']
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
def create_Mortality_Model(train_X, train_Y):
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    model = Sequential()
    model.add(SimpleRNN(16, input_shape=(1, 34)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=16))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=8))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=1))
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
    history = model.fit(train_X, train_Y, validation_split=0.5, epochs=epochsNum, batch_size=100, verbose=2)
    print("Model training is finished")
    metric = ['root_mean_squared_error', 'val_root_mean_squared_error']
    loss = ['loss', 'val_loss']
    display_model_graphs(history, epochsNum, metric, "RMSE", 'Mortality')
    display_model_graphs(history, epochsNum, loss, "RMSE", 'Mortality')
    model.save("..\\Data\\MortalityModel.h5")
    print("Model is saved")


# This function is to re-Train the existing mortality model
def model_Mortality_learning(train_X, train_Y):
    train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    path = "..\\Data\\MortalityModel.h5"
    model = load_model(path)
    adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer=adam, loss='mse', metrics=[keras.metrics.RootMeanSquaredError()])
    model.fit(train_X, train_Y, validation_split=0.5, batch_size=100, epochs=epochsNum, verbose=2)
    model.save("..\\Data\\MortalityModel.h5")
    print("Model is saved")


def MortalityLearning():
    print("start preprocessing")
    train_X, train_Y = gen_Learning_data_Mortality()
    print("done preprocessing")

    print("Starting Mortality Model Learning\n")
    print("----------------------Mortality Model Learning Results----------------------------")
    model_Mortality_learning(train_X, train_Y)
    print("Mortality Model Learning Ended\n")


def main():
    MortalityLearning()


if __name__ == '__main__':
    main()
