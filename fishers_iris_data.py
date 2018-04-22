# Tomás Murray, 2018-04-02
# Course 52167 Programming and Scripting
# Fisher's Iris Data Project

# Raw data file downloaded from https://archive.ics.uci.edu/ml/datasets/iris

# For futher data exploration and manipulation import the pandas library https://pandas.pydata.org/. It is dependent on other libraries such as NumPy http://www.numpy.org/ for additional mathematical functions beyond the Python standard library and matplotlib.pyplot https://matplotlib.org/api/pyplot_api.html to provide a MATLAB-like plotting framework.

import pandas as pd # abbreviate library to simplify code
import numpy as np
import matplotlib.pyplot as plt

# Load raw data as a dataframe

raw = pd.read_csv('data/iris.csv', sep=',')

# Check data loaded correctly and data structure using .head function

print(raw.head())

# Load again, specifying that the data lack column headings and also add them

data = pd.read_csv('data/iris.csv', sep=',', header = None, names = ['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'species'])

# Check data frame is in the correct format

print(data.head())

# Check size of data frame (.shape) and data types per column (.dtypes)

print("The data frame has (rows, columns):", (data.shape))
print(data.dtypes)

# ensure species is being recogised as a categorical as it was originally called 'object' above: https://pandas.pydata.org/pandas-docs/stable/categorical.html 

data['species'] = data['species'].astype('category')
print(data.dtypes)

# Print decriptive statistics about the numerical columns and save as a table in the 'tables' directory

table1 = data.describe()
print(table1)
table1.to_csv('tables/table1.csv')

# Split data into groups by species, print decriptive statistics and save as a table

byspecies = data.groupby('species')
table2 = byspecies.describe()
print(table2)
table2.to_csv('tables/table2.csv')

# Create histograms of features grouped by species. Histograms illustrate the distribution of each feature attributes per species. This aids the identification of the differences, or lack thereof, between species per feature. When exploring differences between classes, I think unstacked histograms per class plotted with the entire distribution in the background more ameniable to interpretation than stacked histograms.

# Originally I tried the below but a bug in histogram generation with matplotlib means that you can't use groupby to automatically assign the labels in the legend. Identical issue on stackoverflow here: https://stackoverflow.com/questions/45883598/pandas-histogram-df-hist-group-by

# Plot histograms of each variable and colour by species
# data.groupby("species").sepalLength.hist(alpha=0.5)
# plt.xlabel('Width (cm)')
# plt.ylabel('Frequency')
# plt.title('Sepal length across species')
# plt.legend()
# plt.show()
# plt.savefig('figures/fig1.jpg')

# Adapting the solution in: https://stackoverflow.com/questions/45883598/pandas-histogram-df-hist-group-by 

# Function to subset each feature (col) by species

def sephist(col):
    setosa = data[data['species'] == 'Iris-setosa'][col]
    versicolor = data[data['species'] == 'Iris-versicolor'][col]
    virginica = data[data['species'] == 'Iris-virginica'][col]
	return setosa, versicolor, virginica

# Produce a list called cols of the column headings to iterate through with enumerate

cols = list(data[['Sepal width (cm)','Sepal length (cm)','Petal width (cm)','Petal length (cm)']])

# for statement to produce four histograms in a facetted 2 x 2 grid.

for num, alpha in enumerate(cols):
    plt.subplot(2, 2, num + 1) # num + 1 missing from original solution
    plt.hist(sephist(alpha)[0], alpha=0.6, label='setosa')
    plt.hist(sephist(alpha)[1], alpha=0.6, label='versicolor')
    plt.hist(sephist(alpha)[2], alpha=0.6, label='virginica')
	plt.legend(loc='upper right')
    plt.title(alpha)
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
plt.savefig('figures/fig1.jpg') # save to figures directory
plt.close() # to clear all features to produce the next figure (I found out the hard way!): https://matplotlib.org/api/_as_gen/matplotlib.pyplot.close.html

# Import the seaborn visualisation library to produce a pairplot. The pairplot will produce stacked histograms (which I dislike) and scatterplots of numerical features, coloured according to Iris species: https://seaborn.pydata.org/generated/seaborn.pairplot.html

# The scatterplots will identify any potential association between features and if these relationships are species-specific.

import seaborn as sns

fig2 = sns.pairplot(data, hue='species', size=2) # hue identifies the class to colourise; size is height (in inches!) of each facet
fig2.fig.subplots_adjust(right = 0.8) # There was a bug in seaborn as the legend is rendered over the pairplot, not outside to the centre right. The solution used on this line to adjust the right margin was posted here: https://stackoverflow.com/questions/37815774/seaborn-pairplot-legend-how-to-control-position

plt.savefig('figures/fig2.jpg')

# To test statistical differences between the species across the measured features, I wanted to repeat the ANOVA analyses in Fisher's original paper but from the figures it was clear that other than Fig. 2, the data was largely non-normal and heteroscedastic so I chose non-parametric tests. I could test for normality too (e.g. Shapiro-Wilk), but with a sample size of 50 across groups there may not be sufficient power to reject normality making the test meaningless.

# Import scipy for Kruskal Wallis test functionality: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html Test the null hypothesis that the distribution of the ranks of the numerical features do not differ across species.

import scipy.stats as sc

# Kruskal Wallis test requires an arrary from the dataframe: https://stackoverflow.com/questions/35276217/use-groups-in-scipy-stats-kruskal-similar-to-r-cran-kruskal-test

def KW(x):
    data.species = np.array(data.species) # convert `data.species` to a numpy array for indexing
    label, idsp = np.unique(x, return_inverse=True) # find unique group labels and their corresponding indices
    groups = [data.species[idsp == i] for i, l in enumerate(label)] # make a list of arrays containing the data.species values corresponding to each unique label
    H, p = sc.kruskal(*groups) # use `*` to unpack the list as a sequence of arguments
    print(H, p)

KW(data['Sepal width (cm)'])
KW(data['Sepal length (cm)'])
KW(data['Petal width (cm)'])
KW(data['Petal length (cm)'])

# Pandas can calculate the Spearman's rho, but does not calculate the p-value (only for Pearons's r). Both rho and p-value can be outputted by the scipy library spearmanr function: https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.spearmanr.html

# Across species correlations. Need to drop species column to change the dataframe with columns of strings to an array to the spearman function in scipy: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html

data.nospecies = data.drop(['species'], axis = 1) # remove species column
rho, pval = sc.spearmanr(data.nospecies) # conduct pairwaise correlation
np.savetxt('tables/table3.csv', rho, delimiter=',') # save rho as a table
np.savetxt('tables/table4.csv', pval, delimiter=',') # save p-value as a table

# Per species correlations.

def correl(y):
    n = str(y['species'].iloc[0]) # get the species name as a string for the output file using pandas syntax for value under the species column at row index 0
	y.nospecies = y.drop(['species'], axis = 1)
    rho, pval = sc.spearmanr(y.nospecies)
    np.savetxt('tables/table_Srho_' + n[5:] + '.csv', rho, delimiter=',') # use + n[5:] + to concatenate output file name with species name and remove 'Iris-'
    np.savetxt('tables/table_Spval_' + n[5:] + '.csv', pval, delimiter=',')

correl(setosa)
correl(versicolor)
correl(virginica)
