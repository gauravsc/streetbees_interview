import pandas as pd 
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt

# Load data into a pandas dataframe
data_df = pd.read_csv('./data/data.csv', sep=',')
labels_df = pd.read_csv('./data/classes.txt', names=['Labels'])

# Create a corresponding column 'Class' (from data.csv) in the dataframe of labels
labels_df['Class'] = np.arange(data_df['Class'].min(), data_df['Class'].max()+1)


#### Data Analysis #### 

# Compute the distribution of various labels and plot it using matplotlin and seaborn
label_cnt = data_df['Class'].value_counts().sort_index()

fig = plt.figure()
sns.set() # switching from default style of matplotlib to sns
ax = sns.barplot(label_cnt.index, label_cnt.values)
ax.set_xlabel('Labels'), ax.set_ylabel('Counts')
fig.savefig('./plots/data_analysis_plots/label_count.png')


# Compute word cloud for each label and plot them