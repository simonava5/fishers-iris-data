# Fisher's Iris Data: Project for GMIT Course 52167 Programming and Scripting

## Table of contents
* [Background information on the data](https://github.com/tomasmurray/fishers-iris-data#background-information-on-the-data)
	* [What is the data](https://github.com/tomasmurray/fishers-iris-data#what-is-the-data)
	* [Who collated it and why](https://github.com/tomasmurray/fishers-iris-data#who-collated-it-and-why)
	* [How data were collected](https://github.com/tomasmurray/fishers-iris-data#how-data-were-collected)
	* [Why it’s still in use today](https://github.com/tomasmurray/fishers-iris-data#why-its-still-in-use-today)
* [Strategy to investigation](https://github.com/tomasmurray/fishers-iris-data#strategy-to-investigation)
	* [Objectives](https://github.com/tomasmurray/fishers-iris-data#objectives)
	* [Data acquistion and formatting](https://github.com/tomasmurray/fishers-iris-data#data-acquistion-and-formatting)
	* [Data exploration](https://github.com/tomasmurray/fishers-iris-data#data-exploration)
	* [Analyses](https://github.com/tomasmurray/fishers-iris-data#analyses)
* [Summary of results](https://github.com/tomasmurray/fishers-iris-data#summary-of-results)
* [References](https://github.com/tomasmurray/fishers-iris-data#references)

## Background information on the data
### What is the data
The Iris flower data set is considered a classic data set for use in statistics [1, 2].  The data set contains 150 objects of three different classes and four features: three species of North American *Iris* flower (*Iris setosa*, *I. virginica* and *I. versicolor*) with 50 numeric values per species of sepal length and width, and petal length and width, in cm.  It is available for download since 1988 from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) [3].

### Who collated it and why
The data were collated by Edgar Anderson for his publication in 1935 to as the species were seen as an ideal study system within which test hypotheses about intra- and interspecific morphological variation, and consequent inference of the evolutionary relationships between them [4, 5]. As *I. versicolor* is geographically and morphologically intermediate between *I. virginica* and *I. setosa*, Anderson aimed to test whether *I. versicolor* originated as a result of a recent hybridisation event between *I. virginica* and *I. setosa*, or whether it was a progenitor of either species [5].  

<p align="center">
  <img alt="Iris species in Fisher's Iris Data" src="https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png">
</p>
<p align="center">
  <b>Figure 1.</b> The three species and parts measured by Edgar in 1935 [4]; image from [6].<br>
</p>

### How data were collected
Two of the species, *I. setosa* and *I. versicolor* were collected on the Gaspé Peninsula, Quebec, Canada "all from the same pasture, and picked on the same day and measured at the same time by the same person with the same apparatus", and published by Edgar in 1935 [1,5].  It is presumed that the same apparatus and rigor were applied to the measurement of *I. virginica* before Edgar shared the data with the British statistician and biologist Sir Ronald Aylmer Fisher in 1936 [7].  

### Why it’s still in use today
Given that it was one of the first widely available and high profile ‘real data’ sets, it’s use proliferated as a pedagogical resource for teaching statistical methodologies [2].  The data were originally used by Fisher as an example for multivariate discriminate analyses [7].  Subsequently, the Irish data set has become one of the most widely used reference data set for classification and prediction studies, and more recently for machine learning approaches to clustering, classification and pattern recognition [8]. 

<p align="center">
  <img alt="Sir Ronald Aylmer Fisher" src="http://www.swlearning.com/quant/kohler/stat/biographical_sketches/Fisher_3.jpeg">
</p>
<p align="center">
  <b>Figure 2.</b> Sir Ronald Aylmer Fisher, British statistician and biologist (1890-1962); image from [9].<br>
</p>

## Strategy to Investigation
### Objectives
* Explore differences between species with regards to the features measured through desciptive statistics, graphs and statistical tests
* Identify relationships between features both across and within species, and test for the strength and statisical significance of these relationships
* Implement a classification approach, identify which features were the most informative and assess the accuracy of the approach

### Data acquistion and formatting
The raw data file was downloaded from UC Irvine Machine Learning Repository [3]. For futher data exploration and manipulation the [pandas](https://pandas.pydata.org/) library was imported. The pandas library is an open-source, [BSD-licensed](http://www.linfo.org/bsdlicense.html) library of functions that greatly facilitate data exploration, manipulation and analysis [10]. It is dependent on other libraries such as [NumPy](http://www.numpy.org/) [11] for additional mathematical functions beyond the Python standard library and [matplotlib.pyplot](https://matplotlib.org/api/pyplot_api.html) [12] module to provide a MATLAB-like plotting framework, both of which must be imported in addition to pandas.

```python
import pandas as pd # abbreviate library to simplify code
import numpy as np
import matplotlib.pyplot as plt
```

The default data structure for pandas is a data frame rather than an array. The raw data were loaded as a data frame and the first five rows examined to check the file format.

```python
raw = pd.read_csv('data/iris.csv', sep=',')
print(raw.head())
```

As the raw data frame lacks column headings, these were added added and the data frame renamed to data. The dimensions of the data frame were checked using the .shape() and the data types per column examined using .dtypes().

```python
data = pd.read_csv('data/iris.csv', sep=',', header = None, names = ['sepalLength', 'sepalWidth', 'petalLength', 'petalWidth', 'species'])
print(data.head())
print("The data frame has (rows, columns):", (data.shape))
print(data.dtypes)
```
<p align="center">
  <img alt="Output of .shape and .dtypes" src="https://github.com/tomasmurray/fishers-iris-data/blob/master/figures/data_frame.PNG">
</p>
<p align="center">
  <b>Figure 3.</b> Output of .shape and .dtypes.<br>
</p>

The numerical features are correctly identified as continuous variables stored as a [double precision float](https://docs.scipy.org/doc/numpy-1.14.0/user/basics.types.html) and the species column as an object. For future exploration and analyses I then specify that the species column represents a categorical variable using the .astype module. 

```python
data['species'] = data['species'].astype('category')
print(data.dtypes)
```

### Data exploration
The pandas [.describe](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html) module generates descriptive statistics for the numerical features of the data frame. This output is then saved as a .csv in the /tables directory.

```python
table1 = data.describe()
print(table1)
table1.to_csv('tables/table1.csv')
```

Statistic |	Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm)
--- | --- | --- | --- | ---
Count |	150 |	150 |	150 |	150
Mean |	5.843 |	3.054 |	3.758 |	1.198
Standard deviaiton	| 0.828 |	0.433 |	1.764 |	0.763
Minimum	| 4.3 |	2 |	1 |	0.1
25% quartile	| 5.1 |	2.8 |	1.6 |	0.3
Median | 5.8 |	3 |	4.35 |	1.3
75% quartile	| 6.4 |	3.3 |	5.1 |	1.8
Maximum	| 7.9 |	4.4 |	6.9 |	2.5

To examine the differences across species, the same summary statistics were generated per species using the .groupby module on the categorical species variable.

```python
byspecies = data.groupby('species')
table2 = byspecies.describe()
print(table2)
table2.to_csv('tables/table2.csv')
```

Speal Length (cm) |	*I. setosa*	| *I. versicolor* |	*I. virginica*
--- | --- | --- | ---
Count |	50 |	50 |	50
Mean |	5.006 |	5.936 |	6.588
Standard deviation |	0.352 |	0.516 |	0.635
Minimum	| 4.3 |	4.9 |	4.9
25% quartile |	4.8 |	5.6 |	6.225
Median |	5	| 5.9 |	6.5
75%	quartile | 5.2 |	6.3 |	6.9
Maximum |	5.8 |	7 |	7.9

Sepal Width (cm) |	*I. setosa*	| *I. versicolor* |	*I. virginica*
--- | --- | --- | ---
Count |	50 |	50 |	50
Mean	| 3.418 |	2.770 |	2.974
Standard deviation |	0.381 |	0.313 |	0.322
Minimum |	2.3 |	2 |	2.2
25% quartile |	3.125 |	2.525 |	2.8
Median |	3.4 |	2.8 |	3
75% quartile |	3.675 |	3 |	3.175
Maximum |	4.4 |	3.4 |	3.8

Petal Length (cm) |	*I. setosa*	| *I. versicolor* |	*I. virginica*
--- | --- | --- | ---
Count |	50 |	50	| 50
Mean	| 1.464 |	4.260 |	5.552
Standard deviation |	0.173 |	0.469 |	0.551
Minimum	| 1 |	3	| 4.5
25% quartile |	1.4 |	4 |	5.1
Median |	1.5 |	4.35 |	5.55
75% quartile |	1.575 |	4.6 |	5.875
Maximum |	1.9 |	5.1 |	6.9

Petal Width (cm) |	*I. setosa*	| *I. versicolor* |	*I. virginica*
--- | --- | --- | ---
Count |	50 |	50 |	50
Mean |	0.244 |	1.326 |	2.026
Standard deviation |	0.107 |	0.197 | 0.274
Minimum |	0.1 |	1 |	1.4
25%	quartile | 0.2 |	1.2 |	1.8
Median |	0.2 |	1.3 |	2
75% quartile |	0.3 |	1.5 |	2.3
Maximum |	0.6 |	1.8 |	2.5

Histograms illustrate the shape of the distribution of each feature per species. This aids the identification of the differences, or lack thereof, between species per feature. When exploring differences between classes, I think unstacked histograms per class plotted with the entire distribution in the background more ameniable to interpretation than stacked histograms.

Initially, I tried to plot a histogram using the pandas .groupby function but this did not produce the expected panel of histograms (identical issue reproted [here](https://stackoverflow.com/questions/45883598/pandas-histogram-df-hist-group-by) on stackoverflow). 

### Analyses
#### Comparison across classes
#### Correlation
#### Classification

## Summary of results

## References

1.	Wikipedia entry: Iris flower data set.
	https://en.wikipedia.org/wiki/Iris_flower_data_set
2.	Singer, J. D., & Willett, J. B. (1990). Improving the teaching of applied statistics: Putting the data back into data analysis. The American Statistician, 44(3), 223-230.
3.	UC Irvine Machine Learning Repository: Iris data set.
	https://archive.ics.uci.edu/ml/datasets/iris
4.	Anderson, E. (1935). The irises of the Gaspe Peninsula. Bulletin of the American Iris society, 59, 2-5.
5.	Anderson, E. (1936). The species problem in Iris. Annals of the Missouri Botanical Garden, 23(3), 457-509.
6.	Datacamp: Machine Learning in R
	https://www.datacamp.com/community/tutorials/machine-learning-in-r
7.	Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of human genetics, 7(2), 179-188.
8.	Runkler, T. A. (2012). Chapter 2: Data and Relations. Models and Algorithms for Intelligent Data Analysis. Vieweg and Teubner Verlag.
9.	Wikipedia entry: Ronald Fisher
	https://en.wikipedia.org/wiki/Ronald_Fisher
10.	pandas library for Python
	https://pandas.pydata.org/
11.	NumPy library for Python
	http://www.numpy.org/
12.	pyplot module of the Matplotib library for Python
	https://matplotlib.org/api/pyplot_api.html
13.	
