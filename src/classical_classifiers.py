import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables 	
data_path = './data/data.csv'
random_nouns_path = './data/nounlist.txt'

classifiers = {
			   'naive_bayes':MultinomialNB(),
			   'linear_classifier': SGDClassifier(),
			   'ridge_classifier': RidgeClassifier()}

def visualize_performance(performance_dict):
	# Plot the frequency count in matplotlib
	fig = plt.figure()
	sns.set() # switching from default style of matplotlib to sns
	ax = sns.barplot(list(performance_dict.keys()), list(performance_dict.values()))
	ax.set_xlabel('Model Type'), ax.set_ylabel('K-fold Accuracy')
	fig.savefig('./plots/model_performance_plots/model_comparison_bar_chart.png')


if __name__ == '__main__':
	data_df = pd.read_csv(data_path, sep=',')
	# data_df = data_df.sample(n=5000, replace=True, random_state=45)

	all_names = data_df['Name'].tolist()
	all_labels = data_df['Class'].tolist()

	# Add random names labelled as label 0 so model can identify OOD names during inference
	random_nouns = pd.read_csv(random_nouns_path, sep=',')
	# random_nouns = random_nouns.sample(n=4000, replace=True, random_state=45)
	all_names.extend(random_nouns['Name'].tolist())
	all_labels.extend([0]*len(random_nouns))

	# Create label to index mapping
	unique_labels = sorted(list(set(all_labels)))
	label2idx = {t: i for i, t in enumerate(unique_labels)}

	# Feature extractor from the text
	vectorizer = TfidfVectorizer()
	X = vectorizer.fit_transform(all_names)
	y = np.array(all_labels)

	print("Shape of X: ", X.shape, ' Shape of Y: ', y.shape)

	accuracy_scorer = make_scorer(accuracy_score)
	performance_dict = {}
	for clf_name, clf_obj in classifiers.items():
		scores = cross_val_score(clf_obj, X, y, cv=5, scoring=accuracy_scorer)
		print("Classifier Name: ", clf_name)
		print("Mean accuracy scores after k-fold cross validation: ", np.mean(scores))
		performance_dict[clf_name] = np.mean(scores)

	visualize_performance(performance_dict)
