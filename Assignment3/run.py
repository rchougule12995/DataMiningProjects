import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth

# Importing pandas library for reading csv files using the read_csv function

original_df_1 = pd.read_csv("./specs/gpa_question1.csv")
# Keeping the original data set in original_df_1 and manipulating with a new data frame to filter the count attribute.

# We will now work on the data set df_1 moving ahead for further operations
df_1 = original_df_1.drop(['count'], axis=1)

# The methods fit() and transform() in TransactionEncoder object requires an input data as a List.
# We will transform the df_1 dataframe to a list add put it in a variable major_list
major_list = df_1.values.tolist()

# Initialization of the TransactionEncoder object in te
te = TransactionEncoder()

# The fit method identifies the unique list of columns/labels in our dataset
# The transform method will transform the List in to a one-hot encoded boolean array
te_ary = te.fit(major_list).transform(major_list)

# In df we get the list of unique columns throughout the data set.
df = pd.DataFrame(te_ary, columns=te.columns_)

# Using the Apriori function we can find the frequent item sets in the data
# The minimum support of 15% can be achieved using the parameter min_support=0.15
frequent_items = apriori(df, min_support=0.15, use_colnames=True)

# Writing the output of the data set to the csv file
frequent_items.to_csv('./output/question1_out_apriori.csv', index=False, header=True)

# The df.apply function applies the function to each row in the data frame
# The lambda function here takes the length of items in the frequent itemset
frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
#print(frequent_items)

# association_rules function is a method of the mlxtend.frequent_patterns
# The metric parameter in the function takes the confidence as parameter.
# we define the threshold value for confidence in min_threshlod = 0.9
df2 = association_rules(frequent_items, metric="confidence", min_threshold=0.9)

# Exporting the output of the df2 to csv file
df2.to_csv('./output/question1_out_rules9.csv', index=False, header=True)

# Similar to the above one the metric parameter in the function takes the confidence as parameter.
# we define the threshold value for confidence in min_threshlod = 0.7
df3 = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

# Exporting the output of the df3 to csv file
df3.to_csv('./output/question1_out_rules7.csv', index=False, header=True)

# *********************************************************************************************************
# Question 2

# Importing the data set from the bank_data_question2
original_df_2 = pd.read_csv("./specs/bank_data_question2.csv")

# print(original_df_2.head())

# Removing the ID attribute and using a different data frame to keep the original dataframe as is
# The ID attribute is unique for all the rows and wont be useful in generation of frequent itemsets
df_4 = original_df_2.drop(['id'], axis=1)

# here is for loop for identifying the numeric data columns in the data frame and binning them for discretization
# we use the cut function here to discretize our numeric data
for i in df_4.columns:
    if df_4[i].dtype == np.int64 or df_4[i].dtype == np.float64:
        df_4[i] = (pd.cut(df_4[i], bins=3))

# We use the get_dummies function from pandas here as we have a mixed dataset of integers/float and string
df_4 = pd.get_dummies(df_4)

# We can then use the dummy coded/binomial output from the getdummies data to the fpgrowth function from mlxtend
# We define minimum support to be 20% with the min_support=0.2 here
frequent_items3 = fpgrowth(df_4, min_support=0.2, use_colnames=True)

# Exporting the frequent item sets from the the data frame to the csv
frequent_items3.to_csv('./output/question2_out_fpgrowth.csv', index=False, header=True)

# We can identify the length of the items in the frequent itemset using the lambda function defined here
frequent_items3['length'] = frequent_items3['itemsets'].apply(lambda x: len(x))

# print(frequent_items3)

# Testing multiple confidence values with different values for the min_threshold parameter in the
# association_rules function
df5 = association_rules(frequent_items3, metric="confidence", min_threshold=0.79)

df5 = association_rules(frequent_items3, metric="confidence", min_threshold=0.7)

#print(df5)

df5.to_csv('./output/question2_out_rules.csv', index=False, header=True)

'''''
# df_5.to_csv("./test.csv")


#print(df_4.head())

'''''