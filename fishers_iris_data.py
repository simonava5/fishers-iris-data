# Tomás Murray, 2018-04-02
# Course 52167 Programming and Scripting
# Fisher's Iris Data Project

# Raw data from https://archive.ics.uci.edu/ml/datasets/iris

# Check raw data frame structure

with open("data/iris.csv") as f:
  for line in f:    
    x = line.split(',')     
    print(('{} {} {} {} {}').format(x[0], x[1], x[2], x[3], x[4]))

# For futher data exploration and manipulation import the pandas library https://pandas.pydata.org/. It is dependent on other libraries too like NumPy http://www.numpy.org/ for additional mathematical functions beyond the Python standard library and matplotlib.pyplot https://matplotlib.org/api/pyplot_api.html to provide a MATLAB-like plotting framework.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load raw data as a dataframe

raw = pd.read_csv('data/iris.csv', sep=',')

# Check data loaded correctly and structure using .head function

print(raw.head())

# Load again, specifying that the data lack column headings and add them

data = pd.read_csv('data/iris.csv', sep=',', header = None, names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'])

# Check data frame is in the correct format

print(data.head())

# Check size of data frame (.shape) and data types per column (.dtypes)

print("The data frame has (rows, columns):", (data.shape))
print(data.dtypes)

# Print decriptive statistics about the numerical columns and save as a table

table1 = data.describe()
print(table1)
table1.to_csv('tables/table1.csv')

# Split data into groups by speices, print decriptive statistics and save as a table

byspecies = data.groupby('species')
table2 = byspecies.describe()
print(table2)
table2.to_csv('tables/table2.csv')

# Originally tried the below but a bug in histogram generation means that you can't use groupby to automatically assign the legend.
# Plot histograms of each variable and colour by species
# data.groupby("species").sepalLength.hist(alpha=0.5)
# plt.xlabel('Width (cm)')
# plt.ylabel('Frequency')
# plt.title('Sepal length across species')
# plt.legend()
# plt.show()
# plt.savefig('figures/fig1.jpg')

# Only straightforward solution is to split data into groups by species for plotting on histograms 

setosa = data[data['species'] == 'Iris-setosa']
versicolor = data[data['species'] == 'Iris-versicolor']
virginica = data[data['species'] == 'Iris-virginica']

# Histogram of sepal length across species

setosa.sepalLength.hist(alpha=0.5, label='setosa')
versicolor.sepalLength.hist(alpha=0.5, label='versicolor')
virginica.sepalLength.hist(alpha=0.5, label='virginica')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Sepal length across species')
plt.legend()
plt.savefig('figures/fig1.jpg')

# use seaborn

import seaborn as sns

sns.pairplot(data, hue='species', size=2)
plt.show()
