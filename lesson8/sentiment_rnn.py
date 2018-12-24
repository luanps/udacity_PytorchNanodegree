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

def pad_features(reviews_ints, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's 
        or truncated to the input seq_length.
    '''
    features = reviews_ints.copy()
    for ii,review in enumerate(reviews_ints):
        length = len(review)
        while length < seq_length:
            features[ii].insert(0,0)
            length += 1
        if length >= seq_length:
            features[ii] = features[ii][:seq_length]
    
    return np.asarray(features)

# Test your implementation!

seq_length = 200

features = pad_features(reviews_ints, seq_length=seq_length)
## test statements - do not change - ##
assert len(features)==len(reviews_ints), "Your features should have as many rows as reviews."
assert len(features[0])==seq_length, "Each feature row should contain seq_length values."
# print first 10 values of the first 30 batches
print(features[:30,:10])

# split data into training, validation, and test data (features and labels, x and y)
split_frac = 0.8
split_idx = int(len(features)*split_frac)
train_x, remaining_x= features[:split_idx], features[split_idx:]
train_y, remaining_y= encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x)*0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))
