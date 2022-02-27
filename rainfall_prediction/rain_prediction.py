# import libraries
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# data cleaning
# read data
data = pd.read_csv("austin_weather.csv")

# drop or delete unnecessary columns in data
data = data.drop(['Events', 'Date', 'SeaLevelPressureHighInches',
                  'SeaLevelPressureLowInches'], axis=1)

# some values have 'T' which denotes no trace of rainfall
# we need to replace all occurences of T with 0 so that
# we can use the data in our model
data = data.replace('T', 0.0)

# the data contains '-' which indicates no or NIL. This means
# that data is not available we need to replace these values
data = data.replace('-', 0.0)

# save data in csv file
data.to_csv('austin_final.csv')
# end of data cleaning

# read the cleaned data
data = pd.read_csv("austin_final.csv")

# the features or the 'x' values of the data these columns are
# used to train the model, the last column, i.e, precipitation
# column will serve as the label
x = data.drop(['PrecipitationSumInches'], axis=1)

# the output or the label
y =data['PrecipitationSumInches']

# reshaping into a 2-D vector and plot a graph to observe this day
day_index = 798
days = [i for i in range(y.size)]

# initialize a linear regression classifier
clf = LinearRegression()

# train the classifier with our input data
clf.fit(x, y)

# give a sample input to test our model this is a 2-D vector that
# contains values for each columns in the dataset.
inp = np.array([[74], [60], [45], [67], [49], [43], [33], [45],
                [57], [29.68], [10], [7], [2], [0], [20], [4], [31]])
inp = inp.reshape(1, -1)

# print the output.
print('The precipitation in inches for the input is:', clf.predict(inp))

# plot a graph of the precipitation levels versus the total number of
# days. one day, which is in red, is tracked. It has precipitation of
# approx. 2 inches.
print("The precipitation trernd grpah:")
plt.scatter(days, y, color='g')
plt.scatter(days[day_index], y[day_index], color='r')
plt.title("Precipitation level")
plt.xlabel("Days")
plt.ylabal("Precipitation in inches")

plt.show()
x_vis = x.filter(['TempAvgF', 'DewPointAvgF', 'HumidityAvgPercent',
                  'SeaLevelPressureAvgInches', 'VisibilityAvgMiles',
                  'WindAvgMPH'], axis=1)

# plot a graph with a few features (x values) against the precipitation
# or rainfall to observe the trends.

print("Precipitation vs selected attributes graph: ")

for i in range(x_vis.columns.size):
    plt.subplot(3, 2, i + 1)
    plt.scatter(days, x_vis[x_vis.columns.values[i][:100]],
                color='g')

    plt.scatter(days[day_index],
                x_vis[x_vis.columns.values[i]][day_index],
                color='r')

    plt.title(x_vis.columns.values[i])

plt.show()