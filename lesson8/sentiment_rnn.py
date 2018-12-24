import numpy as np
from string import punctuation
from collections import Counter
import pdb

# read data from text files
with open('reviews.txt', 'r') as f:
    reviews = f.read()
with open('labels.txt', 'r') as f:
    labels = f.read()

# get rid of punctuation
reviews = reviews.lower() # lowercase, standardize
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by new lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create a list of words
words = all_text.split()

# dictionary that maps words to integers
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {w: ii for ii, w in enumerate(vocab, 1)}

# use the dict to tokenize each review in reviews_split
# store the tokenized reviews in reviews_ints
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# converting labels to 1 and 0
labels = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels])

# remove 0-lenght reviews and their labels
for ii, review in enumerate(reviews_ints):
    if len(review)==0:
        reviews_ints.pop(ii)
        encoded_labels = np.delete(encoded_labels,ii)

