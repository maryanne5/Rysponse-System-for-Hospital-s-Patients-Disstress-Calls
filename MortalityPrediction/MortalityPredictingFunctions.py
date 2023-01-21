import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.python.keras.models import load_model
from Utils.csvFileFunctions import add_column_in_csv, read_csv_data

testSize = 20000


def adjust_Mortality_Data(data, isOneLine):
    # Add more data lines (If we receive one line) so the dummies consideration will be the same as in the learning
    # process (with large data frame)
    if isOneLine:
        data_all = pd.read_csv('..\\Data\\patientsList.csv')
        data = pd.concat([data, data_all], axis=0)
    # Drop columns that we don't need like specific dates, or the id of the patient
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
    # The ultimate goal to perform standardization is to bring down all the features to a common scale without
    # distorting the differences in the range of the values. We use fit_transform() on training data and transform()
    # on the test data
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
    if isOneLine:
        return data[:1]
    return data


def process_Mortality_data(data, lines_number):
    # Save the length of stay in a different variable
    labels = data['Mortality']

    data = adjust_Mortality_Data(data, lines_number == 1)

    # Separate for train and test
    test_X = data.tail(n=lines_number).to_numpy()
    test_Y = labels.tail(n=lines_number).to_numpy()

    return test_X, test_Y

# a method that uses the trained model to predict and return an outcome
# it receives the data which we want to use to predict - (X,Y)
# and a boolean variable that dictates weather we want to predict one data element or more
def model_Mortality_Prediction(test_X, isOneLine):
    path = '..\\Data\\MortalityModel.h5'
    model = load_model(path)
    test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
    pred = model.predict(test_X)

    if isOneLine:
        return pred[0]
    else:
        return pred


# Method for predicting on data element
def PredictMortality(data):
    print("Preprocessing Mortality Data For Testing")
    Test_X, Test_Y = process_Mortality_data(data, 1)
    print("Finished preprocessing")
    print("Begin Mortality Prediction For One Patient\n")
    print("----------------------Prediction Results----------------------------")
    result = model_Mortality_Prediction(Test_X, True)
    print("Prediction Ended\n")
    print("The predicted Mortality is ", result[0])
    return result


# Method that saves the predicted test data in a csv file for later on usage (Decision Tree)
# in the Data folder
# it also displays the Model Evaluation
def Predict_Mortality_Test(data):
    print("Preprocessing Mortality Data For Testing")
    Test_X, Test_Y = process_Mortality_data(data, testSize)
    print("Finished preprocessing")
    print("Begin Mortality Prediction For Testing Batch\n")
    print("----------------------Prediction Results----------------------------")
    pred = model_Mortality_Prediction(Test_X, False)
    print("Moratlity Model Test Evaluation:")
    print("MSE", mean_squared_error(pred, Test_Y))
    print("RMSE", mean_squared_error(pred, Test_Y, squared=False))
    print("MAE", mean_absolute_error(pred, Test_Y))
    add_column_in_csv("..\\Data\\predictedMortality.csv", pred)
    print("Prediction Ended\n")
    print("\n")
    return pred


def main():
    # We will get the data from button interrupt
    data = read_csv_data(False)
    Predict_Mortality_Test(data)


if __name__ == '__main__':
    main()
