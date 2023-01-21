from matplotlib import pyplot as plt


# A method to display a data of a model n a graph
# given the model training history
# number of epochs
# a list containing the metric name for train and val data - [ name, Val_name]
# the metric and models names

def display_model_graphs(history, epochsNum, metric, metricName, ModelName):
    metric_train = history.history[metric[0]]
    metric_val = history.history[metric[1]]
    epochs = range(0, epochsNum)
    plt.plot(epochs, metric_train, 'b', label='Training')
    plt.plot(epochs, metric_val, 'g', label='Validation')
    title = metricName + "of" + ModelName
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metricName)
    plt.legend()
    plt.show()
