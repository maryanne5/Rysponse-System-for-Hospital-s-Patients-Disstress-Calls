import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from LengthOfStayPredicting.LOSPredictingFunctions import read_csv_data


def get_Features():
    data = read_csv_data(False)
    losData = pd.read_csv('..\\Data\\predictedLOS.csv')
    losData = losData[['0']].values
    mortalityData = pd.read_csv('..\\Data\\predictedMortality.csv')
    mortalityData = mortalityData[['0']].values
    urgencyRate = data[['priority rate']].values
    buttonClicks = data[['emergency button clickes amount']].values
    visitsAmount = data[['visits amount']].values
    daysPassed = data[['days passed since last visit']].values
    age = data[['age']].values
    X_train = np.concatenate([age[80000:100000], losData, mortalityData,
                              visitsAmount[80000:100000], daysPassed[80000:100000], buttonClicks[80000:100000]],
                             axis=-1)
    Y_train = urgencyRate[80000:100000]

    return X_train, Y_train


def create_Decision_Tree(X_train, Y_train):
    # Train a decision tree using the activations of the intermediate layers as features
    dt = DecisionTreeRegressor(max_features=6, max_depth=5)
    dt.fit(X_train[:10000], Y_train[:10000])
    pred = dt.predict(X_train[10000:20000])
    RMSE = mean_squared_error(pred, Y_train[10000:20000], squared=False)
    MAE = mean_absolute_error(pred, Y_train[10000:20000])
    R2 = r2_score(pred, Y_train[10000:20000])
    scores = cross_val_score(dt, X_train, Y_train, cv=10)
    # Print the Decision Tree Test and Validation Evaluation
    print("Decision Tree Evaluation:")
    print("Root Mean Squared Error: %0.6f" % RMSE)
    print("Mean Absolute Error: %0.6f" % MAE)
    print("R square coefficient: %0.6f" % R2)
    print("Cross Validation- \n Score: %0.6f \n Standard Deviation: %0.6f" % (scores.mean(), scores.std()))
    return dt


def predict_Patient_priority(data):
    X_train, Y_train = get_Features()
    tree = create_Decision_Tree(X_train, Y_train)
    age = data[['age']].values[0][0]
    lengthofstay = data[['lengthofstay']].values[0][0]
    mortality = data[['Mortality']].values[0][0]
    visitsAmount = data[['visits amount']].values[0][0]
    daysPassed = data[['days passed since last visit']].values[0][0]
    clickesAmount = data[['emergency button clickes amount']].values[0][0]
    features = [age, lengthofstay, mortality, visitsAmount, daysPassed, clickesAmount]
    predictedUrgency = tree.predict([features])
    res = predictedUrgency[0]
    res = int(res)
    return res


#The main is used to check the usage of the decision tree]
# The UI Calls The Above Method - "predict_Patient_priority" to predict the patients priority rate

# def predict_Patient_priority_testing(dt, features):
#     predictedUrgency = dt.predict([features])
#     return int(predictedUrgency)
#
#
# def main():
#     # example of usage
#     X_train, Y_train = get_Features()
#     clf = create_Decision_Tree(X_train, Y_train)
#     print(predict_Patient_priority_testing(clf, [49.6, 5, 0.63, 3, 173, 4]))
#     print(predict_Patient_priority_testing(clf, [54.8, 20, 0.57, 8, 0, 0]))
#
#
# if __name__ == '__main__':
#     main()
