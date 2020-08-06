import argparse
import pandas as pd 
import seaborn as sns 
import numpy as np
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect
import matplotlib.pyplot as plt


#######################
#### Data Analysis ####
#######################

# Compute the distribution of various labels and plot it using matplotlin and seaborn
def explore_label_dsitribution(data_df, labels_df):
	label_cnt = data_df['Class'].value_counts().sort_index()

	# Plot the frequency count in matplotlib
	fig = plt.figure()
	sns.set() # switching from default style of matplotlib to sns
	ax = sns.barplot(label_cnt.index, label_cnt.values)
	ax.set_xlabel('Labels'), ax.set_ylabel('Counts')
	fig.savefig('./plots/data_analysis_plots/label_count_bar_chart.png')


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
def explore_using_word_clouds(data_df, labels_df):
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
def explore_lang_distribution(data_df, labels_df):
	lang_cnt = {}
	for val in data_df['Name'].sample(n=50000, random_state=45):
		val = str(val)
		try:
			lang = detect(val)
		except:
			continue

		if lang in lang_cnt:
			lang_cnt[lang] += 1
		else:
			lang_cnt[lang] = 1

	fig = plt.figure()
	sns.set() # switching from default style of matplotlib to sns
	ax = sns.barplot(list(lang_cnt.keys()), list(lang_cnt.values()))
	ax.set_xlabel('Lang'), ax.set_ylabel('Sample count')
	fig.savefig('./plots/data_analysis_plots/language_count_bar_chart.png')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--label_distr', type=int, default=1)
	parser.add_argument('--word_clouds', type=int, default=1)
	parser.add_argument('--lang_distr', type=int, default=1)
	args = parser.parse_args()

	print ('started loading data from files ...')

	# Load data into a pandas dataframe
	data_df = pd.read_csv('./data/data.csv', sep=',')
	labels_df = pd.read_csv('./data/classes.txt', names=['Labels'])

	# Create a corresponding column 'Class' (from data.csv) in the dataframe of labels
	labels_df['Class'] = np.arange(data_df['Class'].min(), data_df['Class'].max()+1)

	print ('done loading all data from files ...')

	# Check what explorations the user wanted to perform
	if args.label_distr:
		explore_label_dsitribution(data_df, labels_df)
	if args.word_clouds:
		explore_using_word_clouds(data_df, labels_df)
	if args.lang_distr:
		explore_lang_distribution(data_df, labels_df)

