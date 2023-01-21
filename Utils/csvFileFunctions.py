import pandas as pd


# create a csv file with data received as a list
def add_column_in_csv(path, data):
    data_new = pd.DataFrame(data)
    data_new.to_csv(path)


# read data from csv file
def read_csv_data(isOneLine):
    # Create dataframe from csv
    data = pd.read_csv('..\\Data\\patientsList.csv')
    if isOneLine:
        return data[:1]
    return data
