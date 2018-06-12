"""Read, split and save the Tweet dataset for our model"""

import os
import sys


def load_dataset(path_file):
	"""Loads dataset into memory from file"""
	# Open the file, need to specify the encoding for python3
	dataset = []
	use_python3 = sys.version_info[0] >= 3
	with (open(path_file, encoding="utf-8") if use_python3 else open(path_file)) as f:
		# Each line of the file corresponds to one tweet
		for line in f.read().splitlines():
			line = line.split("\t")
			if len(line) <= 1:
				continue
			tag = line[-1]
			if tag != '1' and tag != '0':
				continue
			if len(line) > 2:
				line = "\t".join(line[:-1])
			tweet = line[0].split(" ")
			dataset.append((tweet, tag))
	return dataset


def save_dataset(dataset, save_dir):
	"""Writes tweets.txt and labels.txt files in save_dir from dataset

	Args:
		dataset: ([(["a", "cat"], "O"), ...])
		save_dir: (string)
	"""
	# Create directory if it doesn't exist
	print("Saving in {}...".format(save_dir))
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	# Export the dataset
	with open(os.path.join(save_dir, 'tweets.txt'), 'w') as file_tweets:
		with open(os.path.join(save_dir, 'labels.txt'), 'w') as file_labels:
			for words, tag in dataset:
				file_tweets.write("{}\n".format(" ".join(words)))
				file_labels.write("{}\n".format(tag))
	print("- done.")


if __name__ == "__main__":
	# Check that the dataset exists 
	path_dataset = 'data/tweets_text/dataset'
	msg = "{} file not found. Make sure you have downloaded the right dataset".format(path_dataset)
	assert os.path.isfile(path_dataset), msg

	# Load the dataset into memory
	print("Loading Harvard/CNN dataset into memory...")
	dataset = load_dataset(path_dataset)
	print("- done.")

	# Split the dataset into train, val and split (dummy split, the shuffling was done before)
	train_dataset = dataset[:int(0.8*len(dataset))]
	val_dataset = dataset[int(0.8*len(dataset)) : int(0.9*len(dataset))]
	test_dataset = dataset[int(0.9*len(dataset)):]

	# Save the datasets to files
	save_dataset(train_dataset, 'data/tweets_text/train')
	save_dataset(val_dataset, 'data/tweets_text/val')
	save_dataset(test_dataset, 'data/tweets_text/test')