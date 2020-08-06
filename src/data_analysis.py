import pandas as pd 
import seaborn as sns 
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect
import matplotlib.pyplot as plt


# Load data into a pandas dataframe
data_df = pd.read_csv('./data/data.csv', sep=',')
labels_df = pd.read_csv('./data/classes.txt', names=['Labels'])

# Create a corresponding column 'Class' (from data.csv) in the dataframe of labels
labels_df['Class'] = np.arange(data_df['Class'].min(), data_df['Class'].max()+1)


#######################
#### Data Analysis ####
#######################

# Compute the distribution of various labels and plot it using matplotlin and seaborn
label_cnt = data_df['Class'].value_counts().sort_index()

# Plot the frequency count in matplotlib
fig = plt.figure()
sns.set() # switching from default style of matplotlib to sns
ax = sns.barplot(label_cnt.index, label_cnt.values)
ax.set_xlabel('Labels'), ax.set_ylabel('Counts')
fig.savefig('./plots/data_analysis_plots/label_count.png')

# Function to obtain word cloud of names
def create_word_cloud(df):
	comment_words = '' 
	stopwords = set(STOPWORDS)

	# Iterate through the csv file
	for val in df['Name']:
		val = str(val)
		tokens = val.split()
		
		for i in range(len(tokens)):
			tokens[i] = tokens[i].lower()

		comment_words += " ".join(tokens)+" "

	wordcloud = WordCloud(
		width=1600, height=1600, 
		background_color ='white', 
        stopwords = stopwords, 
        min_font_size = 40).generate(comment_words)

	return wordcloud

# Compute word cloud for each label and plot them
labels = labels_df['Labels'].values
for i in range(1, 15):
	fig = plt.figure()
	temp_df = data_df[data_df['Class'] == i]
	ax1 = fig.add_subplot(1,1,1)
	wordcloud = create_word_cloud(temp_df)
	ax1.imshow(wordcloud)
	ax1.set_title(labels[i-1])
	plt.axis("off") 
	plt.tight_layout(pad = 1.00)
	fig.savefig('./plots/data_analysis_plots/word_cloud_plots/'+labels[i-1]+'.png')
	# fig.savefig('./plots/data_analysis_plots/word_clouds.pdf')



# Compute various languages of the text

