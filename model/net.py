"""Defines the neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

'''
# addition to know when to stop
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
# additions for GloVe
import torchtext.vocab as vocab

# 6B for Wikipedia: 400K vocab, uncased, 50d,100d,200d & 300d vectors, 822 MB
# twitter.27B for Twitter: 1.2M vocab, uncased, 25d, 50d, 100d & 200d vectors, 1.52 GB
glove = vocab.GloVe(name='twitter.27B', dim=100)
'''

class Net(nn.Module):
	"""
	We choose the components (e.g. LSTMs, linear layers etc.) of the network in the __init__ function. We then apply these layers
	on the input step-by-step in the forward function. We use torch.nn.functional to apply functions
	such as F.relu, F.sigmoid, F.softmax. 
	"""

	def __init__(self, params):
		"""
		We define an recurrent network that predicts the label for each tweet. The components
		required are:

		- an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
		- lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
		- fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

		Args:
			params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
		"""
		super(Net, self).__init__()

		# the embedding takes as input the vocab_size and the embedding_dim
		self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim)

		# the LSTM takes as input the size of its input (embedding_dim), its hidden size
		# for more details on how to use it, check out the documentation
		self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True, bidirectional=True)

		# the fully connected layer transforms the output to give the final output layer
		self.fc = nn.Linear(params.lstm_hidden_dim * 2, 1)
		
	def forward(self, s):
		"""
		This function defines how we use the components of our network to operate on an input batch.

		Args:
			s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
			   the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
			   tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
			   the token in the vocab.

		Returns:
			out: (Variable) dimension batch_size*seq_len x 1 with the activations of tokens for each token
				 of each sentence.

		Note: the dimensions after each step are provided
		"""
		#                                -> batch_size x seq_len
		# apply the embedding layer that maps each token to its embedding
		s = self.embedding(s)            # dim: batch_size x seq_len x embedding_dim
		
		# run the LSTM along the sentences of length seq_len
		s, _ = self.lstm(s)              # dim: batch_size x seq_len x lstm_hidden_dim

		# make the Variable contiguous in memory (a PyTorch artefact)
		s = s.contiguous()

		# reshape the Variable so that each row contains one token
		s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim

		# apply the fully connected layer and obtain the output (before softmax) for each token
		s = self.fc(s)                   # dim: batch_size*seq_len x 1

		# apply logistic regression on each token's output 
		return F.sigmoid(s) # dim: batch_size*seq_len x 1


def loss_fn(outputs, labels):
	"""
	Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
	for PADding tokens.

	Args:
		outputs: (Variable) dimension batch_size*seq_len x 1 - logistic regression output of the model
		labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1],
				or -1 in case it is a PADding token.

	Returns:
		loss: (Variable) cross entropy loss for all tokens in the batch

	Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
		  demonstrates how you can easily define a custom loss function.
	"""
	batch_size = labels.shape[0]
	seq_len = labels.shape[1]

	# reshape labels to give a flat vector of length batch_size*seq_len
	labels = labels.view(-1)



	# since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
	mask = (labels >= 0).float()

	# indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
	# number. This does not affect training, since we ignore the PADded tokens with the mask.
	labels = labels % 2

	return torch.norm((outputs[range(outputs.shape[0]), 0] - labels)* mask)
	
def accuracy(outputs, labels):
	"""
	Compute the accuracy, given the outputs and labels for all words. Exclude PADding terms.

	Args:
		outputs: (np.ndarray) dimension batch_size*seq_len x 2 - logistic regression output of the model
		labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
				[0, 1], or -1 in case it is a PADding token.

	Returns: (float) accuracy in [0,1]
	"""
	batch_size = labels.shape[0]
	seq_len = labels.shape[1]

	# reshape labels to give a flat vector of length batch_size*seq_len
	labels = labels.ravel()

	# since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
	mask = (labels >= -0)

	correct = 0.0
	for i in range(batch_size):
		indiv_mask = mask[i*seq_len:(i+1)*seq_len]
		num_words = sum(indiv_mask)
		average = np.sum(outputs[i*seq_len:(i+1)*seq_len, 0] * indiv_mask)/num_words

		prediction = int(average > 0.5)

		if prediction == labels[i*seq_len].item():
			correct += 1.0

	return correct / batch_size

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
	'accuracy': accuracy
}
