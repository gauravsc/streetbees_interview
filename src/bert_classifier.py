import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from model.bert_sequence_classifier import BertForMultiClassSequenceClassification
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Global variables
MAX_LEN = 10
bs = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
data_file_path = './data/data.csv'
epochs = 3
max_grad_norm = 1.0
FULL_FINETUNING = True



if __name__ == '__main__':

	# Instantiate the bert pretrained tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# Instantiate the model
	model = BertForMultiClassSequenceClassification(num_labels=14)

	print ('started loading data from files ...')
	data_df = pd.read_csv(file_path, sep=',')
	print ('done loading all data from files ...')
	
	all_names = data_df['Name'].tolist()
	all_labels = data_df['Class'].values

	unique_labels = [0]+sorted(list(set(all_labels)))
	label2idx = {t: i for i, t in enumerate(unique_labels)}

	tokenized_names = [tokenizer.tokenize('[CLS] ' + name + ' [SEP]') for name in all_names]
	input_ids = [tokenizer.convert_tokens_to_ids(tokenized_name) for tokenized_name in tokenized_names]

	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", value=0.0, truncating="post", padding="post")
	attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

	tr_inputs, val_inputs, tr_labels, val_labels = train_test_split(input_ids, all_labels, random_state=2018, test_size=0.1)
	tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2018, test_size=0.1)

	tr_inputs = torch.tensor(tr_inputs)
	val_inputs = torch.tensor(val_inputs)
	tr_labels= torch.tensor(tr_labels)
	val_labels = torch.tensor(val_labels)
	tr_masks = torch.tensor(tr_masks)
	val_masks = torch.tensor(val_masks)

	# Initiating random sampler for model training
	train_data = TensorDataset(tr_inputs, tr_masks, tr_labels)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

	valid_data = TensorDataset(val_inputs, val_masks, val_labels)
	valid_sampler = SequentialSampler(valid_data)
	valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

	# Total number of training steps is number of batches * number of epochs.
	total_steps = len(train_dataloader) * epochs

	# Whether entire model needs to be fine-tuned or only classification layer
	if FULL_FINETUNING:
	    param_optimizer = list(model.named_parameters())
	    no_decay = ['bias', 'gamma', 'beta']
	    optimizer_grouped_parameters = [
	        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
	         'weight_decay_rate': 0.01},
	        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
	         'weight_decay_rate': 0.0}
	    ]
	else:
	    param_optimizer = list(model.classifier.named_parameters())
	    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

	# Define the adams optimizer
	optimizer = AdamW(
	    optimizer_grouped_parameters,
	    lr=3e-5,
	    eps=1e-8
	)

	# Create the learning rate scheduler.
	scheduler = get_linear_schedule_with_warmup(
	    optimizer,
	    num_warmup_steps=0,
	    num_training_steps=total_steps
	)


	# Code for training the model
	# Store the average loss after each epoch so we can plot them
	loss_values, validation_loss_values = [], []
	for _ in trange(epochs, desc='Epochs'):

		# Put the model in training mode
		model.train()
		# Set the total loss for the current epoch
		total_loss = 0

		####################
		####  TRAINING  ####
		####################

		# Training loop
		for step, batch in enumerate(train_dataloader):
			# Treansfer batch to the GPU
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_masks, b_labels = batch

			model.zero_grad()
			loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_masks, labels=b_labels)
			loss.backward()
			total_loss += loss.item()
			torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
			optimizer.step()
        	scheduler.step()

        # Calculate the average loss over the training data.
    	avg_train_loss = total_loss / len(train_dataloader)
    	print("Average train loss: {}".format(avg_train_loss))

    	# Store the loss value for plotting the learning curve.
    	loss_values.append(avg_train_loss)

    	####################
		###  VALIDATION  ###
		####################

		# Validation loop after completion of a training epoch
		for step, batch in enumerate(valid_dataloader):
			# Reset the validation loss for this epoch.
		    eval_loss, eval_accuracy = 0, 0
		    nb_eval_steps, nb_eval_examples = 0, 0
		    predictions , true_labels = [], []

		    for batch in valid_dataloader:
		        batch = tuple(t.to(device) for t in batch)
		        b_input_ids, b_input_mask, b_labels = batch

		        with torch.no_grad():
		            # Forward pass, calculate logit predictions.
		            # This will return the logits rather than the loss because we have not provided labels.
		            loss, logits = model(b_input_ids, token_type_ids=None,
		                            attention_mask=b_input_mask, labels=b_labels)

	            # Move logits and labels to CPU
		        logits = logits.detach().cpu().numpy()
		        label_ids = b_labels.to('cpu').numpy()

		        # Calculate the accuracy for this batch of test sentences.
		        eval_loss += loss.mean().item()
		        predictions.extend([list(p) for p in np.argmax(logits, axis=1)])
		        true_labels.extend(label_ids)

			    # Computing loss for the batch
			    eval_loss = eval_loss / len(valid_dataloader)
			    validation_loss_values.append(eval_loss)
			    print("Validation loss: {}".format(eval_loss))
			    pred_labels = [unique_labels[p_i] for p, l in zip(predictions, true_labels)
			                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
			    valid_labels = [unique_labels[l_i] for l in true_labels
			                                  for l_i in l if tag_values[l_i] != "PAD"]
			    print("Validation Accuracy: {}".format(accuracy_score(pred_labels, valid_label)))
			    print("Validation F1-Score: {}".format(f1_score(pred_labels, valid_label)))
			    print()
