# Tomás Murray, 2018-04-02
# Course 52167 Programming and Scripting
# Fisher's Iris Data Project

# Raw data from https://archive.ics.uci.edu/ml/datasets/iris

# Check raw data frame structure as originally taught for this data file

with open("data/iris.csv") as f:
  for line in f:    
    x = line.split(',')     
    print(('{} {} {} {} {}').format(x[0], x[1], x[2], x[3], x[4]))

# For futher data exploration and manipulation import the pandas library https://pandas.pydata.org/. It is dependent on other libraries such as NumPy http://www.numpy.org/ for additional mathematical functions beyond the Python standard library and matplotlib.pyplot https://matplotlib.org/api/pyplot_api.html to provide a MATLAB-like plotting framework.

import pandas as pd # abbreviate library to simplify code
import numpy as np
import matplotlib.pyplot as plt

# Load raw data as a dataframe

raw = pd.read_csv('data/iris.csv', sep=',')

# Check data loaded correctly and data structure using .head function

print(raw.head())

# Load again, specifying that the data lack column headings and also add them

data = pd.read_csv('data/iris.csv', sep=',', header = None, names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'])

# Check data frame is in the correct format

print(data.head())

# Check size of data frame (.shape) and data types per column (.dtypes)

print("The data frame has (rows, columns):", (data.shape))
print(data.dtypes)

# Print decriptive statistics about the numerical columns and save as a table in the 'tables' folder

table1 = data.describe()
print(table1)
table1.to_csv('tables/table1.csv')

# Split data into groups by speices, print decriptive statistics and save as a table

byspecies = data.groupby('species')
table2 = byspecies.describe()
print(table2)
table2.to_csv('tables/table2.csv')

# Create histograms of features grouped by species. Histograms illustrate the distribution of each feature attributes per species. This aids the identification of the differences, or lack thereof, between species per feature. When exploring differences between classes, I think unstacked histograms per class plotted with the entire distribution in the background more ameniable to interpretation than stacked histograms.

# Originally I tried the below but a bug in histogram generation with matplotlib means that you can't use groupby to automatically assign the labels in the legend.

# Plot histograms of each variable and colour by species
# data.groupby("species").sepalLength.hist(alpha=0.5)
# plt.xlabel('Width (cm)')
# plt.ylabel('Frequency')
# plt.title('Sepal length across species')
# plt.legend()
# plt.show()
# plt.savefig('figures/fig1.jpg')

# Only straightforward solution is to split data by species for plotting on histograms 

setosa = data[data['species'] == 'Iris-setosa']
versicolor = data[data['species'] == 'Iris-versicolor']
virginica = data[data['species'] == 'Iris-virginica']

# I know this violates the 'Don't Repeat Yourself' principle, but with differing axes and titles I didn't think it a good use of time to write a function for just four graphs. 

# Fig.1 Histogram of sepal length across species. Based on: https://stackoverflow.com/questions/21548750/plotting-histograms-against-classes-in-pandas-matplotlib/21549391  

data.sepalLength.hist(alpha=0.1, label='all species') # alpha = transparency
setosa.sepalLength.hist(alpha=0.6, label='setosa')
versicolor.sepalLength.hist(alpha=0.6, label='versicolor')
virginica.sepalLength.hist(alpha=0.6, label='virginica')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Sepal length across species')
plt.legend()
plt.savefig('figures/fig1.jpg')
plt.close() # to clear all features to produce the next figure (I found out the hard way!): https://matplotlib.org/api/_as_gen/matplotlib.pyplot.close.html

# Fig.2 Histogram of sepal width across species.

data.sepalWidth.hist(alpha=0.1, label='all species')
setosa.sepalWidth.hist(alpha=0.6, label='setosa')
versicolor.sepalWidth.hist(alpha=0.6, label='versicolor')
virginica.sepalWidth.hist(alpha=0.6, label='virginica')
plt.xlabel('Width (cm)')
plt.ylabel('Frequency')
plt.title('Sepal width across species')
plt.legend()
plt.savefig('figures/fig2.jpg')
plt.close()

# Fig.3 Histogram of petal length across species.

data.petalLength.hist(alpha=0.1, label='all species')
setosa.petalLength.hist(alpha=0.6, label='setosa')
versicolor.petalLength.hist(alpha=0.6, label='versicolor')
virginica.petalLength.hist(alpha=0.6, label='virginica')
plt.xlabel('Length (cm)')
plt.ylabel('Frequency')
plt.title('Petal length across species')
plt.legend()
plt.savefig('figures/fig3.jpg')
plt.close()

# Fig.4 Histogram of petal width across species.

data.petalWidth.hist(alpha=0.1, label='all species')
setosa.petalWidth.hist(alpha=0.6, label='setosa')
versicolor.petalWidth.hist(alpha=0.6, label='versicolor')
virginica.petalWidth.hist(alpha=0.6, label='virginica')
plt.xlabel('Width (cm)')
plt.ylabel('Frequency')
plt.title('Petal width across species')
plt.legend()
plt.savefig('figures/fig4.jpg')
plt.close()

# Import the seaborn visualisation library to produce a pairplot. The pairplot will produce stacked histograms (which I dislike) and scatterplots of numerical features, coloured according to Iris species: https://seaborn.pydata.org/generated/seaborn.pairplot.html

# The scatterplots will identify any potential correlations between features and if these relationships are species-specific.

import seaborn as sns

fig5 = sns.pairplot(data, hue='species', size=2) # hue identifies the class to colourise; size is height (in inches!) of each facet
fig5.fig.subplots_adjust(right = 0.8) # There was a bug in seaborn as the legend is rendered over the pairplot, not outside to the centre right. The solution used on this line to adjust the right margin was posted here: https://stackoverflow.com/questions/37815774/seaborn-pairplot-legend-how-to-control-position
plt.savefig('figures/fig5.jpg')
