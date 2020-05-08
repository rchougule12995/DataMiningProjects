import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# Importing the data set SensorData_question1
df = pd.read_csv("./specs/SensorData_question1.csv")

# Test printing the first elements of the data set
# print(df.head())

# Task 1
"""----------------------------------------------------------------------------------------------------------------"""
# Inserting data from column Input3 to OriginalInput3
df.insert((len(df.columns)), "Original Input3", df["Input3"].round(3), allow_duplicates=True)

# Inserting data from column Input12 to OriginalInput12
df.insert((len(df.columns)), "Original Input12", df["Input12"].round(3), allow_duplicates=True)

"""----------------------------------------------------------------------------------------------------------------"""
# Task 2

# Z score can be calculated using mean and standard deviation for the column Input3

# Zscore_calc variable for getting input column
zscore_calc = df["Input3"]

# Zscore calculation using the formula
zscore = (zscore_calc - zscore_calc.mean())/zscore_calc.std()

# Assigning the zscore value to the column df["Input3"]
df["Input3"] = zscore
"""----------------------------------------------------------------------------------------------------------------"""
# Task 3

# Using MinMax Normalisation to normalise the attribute to range 0 to 1

# calculate min value in Input 12 column
df_min = df["Input12"].min()

# calculate max value in Input 12 column
df_max = df["Input12"].max()

# Loop for calculating the Normalized value in Input12 using the minmax formula
for index, row in df.iterrows():
    row["Input12"] = (row["Input12"] - df_min) / (df_max - df_min)

print(df.head())

"""----------------------------------------------------------------------------------------------------------------"""
# Task 4

# Calculating the average for first 12 columns

df["Average Input"] = df.iloc[:0:12].mean(axis=1)

# Writing the csv output to the file
df.to_csv('output/question1_out.csv', index=False, header=True)

"""----------------------------------------------------------------------------------------------------------------"""

# Question 2

# Task 1
"""----------------------------------------------------------------------------------------------------------------"""

# Reading the data values from the input csv
df2 = pd.read_csv("./specs/DNAData_question2.csv")

# keeping a copy of the original dataframe as we need to use it later on
df2_original = df2

# PCA function from the sklearn.decomposition.PCA library which reduces the dimensionality of the data
pca = PCA(0.95)

# The pca.fit_transform will give us the reduced dimension data in the form of an n-d array
test = pca.fit_transform(df2)

# print(type(test))

# Task 2: Binning PCA generated attributes with bins of equal width
"""----------------------------------------------------------------------------------------------------------------"""
# We need to convert this test variable into 1-d array

# cut and qcut works on 1d array.

pca_x = []

# loop each component of pca_compenent into q_cut

for i in range(22):
    # pca_test variable will give us the subset of the discretised data in the bins using function qcut() from pandas
    pca_test = pd.qcut(test[:, i], 10, labels=False)
    pca_x.append(pca_test)
    # Appending the pca_test bins to the list pca_x

# Adding the list to an array as we need to append the data to original data frame
pca_x_arr = np.asarray(pca_x)

# The dimension of the array was 22 * 88, hence we need to take a transpose of the array to append it to the data frame
pca_x_arr1 = pca_x_arr.T

# print(pca_x_arr1.shape)

# Creating a pca_df data frame to merge it with the orignial data frame
pca_df = pd.DataFrame(pca_x_arr1)

# Appending the binned dataset columns to the original data frame
df2_original = pd.concat([df2_original, pca_df], axis=1)

for i in range(22):
    df2_original.rename(columns={i: 'pca' + str(i) + '_width'}, inplace=True)

# Task 3: Binning PCA generated attributes with bins of equal frequency
"""----------------------------------------------------------------------------------------------------------------"""

pca_x2 = []

for i in range(22):
    # pca_test variable will give us the subset of the discretised data in the bins using the function cut() from pandas
    pca_test2 = pd.cut(test[:, i], 10, labels=False)
    pca_x2.append(pca_test2)
    # Appending the pca_test bins to the list pca_x

# Adding the list to an array as we need to append the data to original data frame
pca_x_arr2 = np.asarray(pca_x2)

# The dimension of the array was 22 * 88, hence we need to take a transpose of the array to append it to the data frame
pca_x_arr11 = pca_x_arr2.T

# print(pca_x_arr1.shape)

# Creating a pca_df data frame to merge it with the orignial data frame
pca_df2 = pd.DataFrame(pca_x_arr11)

# Appending the binned dataset columns to the original data frame
df2_original = pd.concat([df2_original, pca_df2], axis=1)

for i in range(22):
    df2_original.rename(columns={i: 'pca' + str(i) + '_freq'}, inplace=True)

df2_original.to_csv('output/question2_out.csv', index=False, header=True)

