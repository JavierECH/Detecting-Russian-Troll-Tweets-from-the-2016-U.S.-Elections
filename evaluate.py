"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
import utils
import model.net as net
from model.data_loader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/tweets_text', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
					 containing weights to load")

def verbosePredict(labels, outputs, out):
	mask = (labels >= 0.0)
	outputs = [output[0] for output in outputs] * mask
	num_words = sum(mask)

	average = float(sum(outputs)) / num_words

	yy = int(average > 0.5)
	y = labels[0]

	out.write('Truth: %s, Prediction: %s [%s]\n' % (y, yy, 'CORRECT' if y == yy else 'WRONG'))
	outputs = [round(output, 4) for output in outputs if output != 0.0]
	out.write(str(outputs) + '\n')



def outputErrorAnalysis(data_batch, labels_batch, output_batch, batch_size, out):
	data_batch = data_batch.data.cpu().numpy()

	args = parser.parse_args()
	vocab_path = os.path.join(args.data_dir, 'words.txt')
	vocab = []
	with open(vocab_path) as f:
		for i, l in enumerate(f.read().splitlines()):
			vocab.append(l)
	max_len = (output_batch.shape[0] + 1) // batch_size

	data_batch_words = []
	for tweet in data_batch:
		tweet_words = []
		for index in tweet:
			tweet_words.append(vocab[index])
		data_batch_words.append(tweet_words)
	data_batch = data_batch_words


	for i in range(batch_size): # batch_size
		tweet = [word for word in data_batch[i] if word != '<pad>']
		out.write('=== ' + ' '.join(tweet) + '\n')
		verbosePredict(labels_batch[i], output_batch[i * max_len:(i+1)*max_len], out)


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps):
	"""Evaluate the model on `num_steps` batches.

	Args:
		model: (torch.nn.Module) the neural network
		loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
		data_iterator: (generator) a generator that generates batches of data and labels
		metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
		params: (Params) hyperparameters
		num_steps: (int) number of batches to train on, each of size params.batch_size
	"""

	# set model to evaluation mode
	model.eval()

	# summary for current eval loop
	summ = []

	# comment out during training
	out = open(os.path.join(args.data_dir, 'test/error-analysis'), 'w')

	# compute metrics over the dataset
	for _ in range(num_steps): # num_steps
		# fetch the next evaluation batch
		data_batch, labels_batch = next(data_iterator)
		
		# compute model output
		output_batch = model(data_batch)

		loss = loss_fn(output_batch, labels_batch)

		# extract data from torch Variable, move to cpu, convert to numpy arrays
		output_batch = output_batch.data.cpu().numpy()
		labels_batch = labels_batch.data.cpu().numpy()

		# comment out during training	
		outputErrorAnalysis(data_batch, labels_batch, output_batch, params.batch_size, out)

		# compute all metrics on this batch
		summary_batch = {metric: metrics[metric](output_batch, labels_batch)
						 for metric in metrics}
		summary_batch['loss'] = loss.item()
		summ.append(summary_batch)

	# cpmment out during training
	out.close()
	# compute mean of all metrics in summary
	metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
	metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
	logging.info("- Eval metrics : " + metrics_string)
	return metrics_mean


if __name__ == '__main__':
	"""
		Evaluate the model on the test set.
	"""
	# Load the parameters
	args = parser.parse_args()
	json_path = os.path.join(args.model_dir, 'params.json')
	assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
	params = utils.Params(json_path)

	# use GPU if available
	params.cuda = torch.cuda.is_available()     # use GPU is available

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: torch.cuda.manual_seed(230)
		
	# Get the logger
	utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

	# Create the input data pipeline
	logging.info("Creating the dataset...")

	# load data
	data_loader = DataLoader(args.data_dir, params)
	data = data_loader.load_data(['test'], args.data_dir)
	test_data = data['test']

	# specify the test set size
	params.test_size = test_data['size']
	test_data_iterator = data_loader.data_iterator(test_data, params)

	logging.info("- done.")

	# Define the model
	model = net.Net(params).cuda() if params.cuda else net.Net(params)
	
	loss_fn = net.loss_fn
	metrics = net.metrics
	
	logging.info("Starting evaluation")

	# Reload weights from the saved file
	utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

	# Evaluate
	num_steps = (params.test_size + 1) // params.batch_size
	test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)
	save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
	utils.save_dict_to_json(test_metrics, save_path)
