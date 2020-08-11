## Installation

```shell
cd streetbees_nlp_exercise/
pip install -r requirements.txt
```



## Execute code

#### Data analysis

```shell
python src/data_analysis.py
```

Data analysis plots are saved in the directory  ./plots/data_analysis_plots 

#### Execute deep learning classifier

```shell
python src/bert_classifier.py 
```

This script would start a bert based classifier and print the accuracy of the model on the screen, in addition to drawing the plots and saving them in the directory './plots/model_train_plots'

#### Execute classical (non-DL) classifiers

```
python src/classical_classifiers.py
```

This script would run a number of classical machine learning models, such as linear classifiers, naive bayes classifier, etc., and output accuracy of every classifier after 5-fold cross-validation, along with plotting a bar chart of model performances in the directory: './plots/model_performance_plots'



