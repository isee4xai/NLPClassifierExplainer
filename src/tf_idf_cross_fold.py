

# numpy
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
stops = set(stopwords.words('english'))

        
def tokenizer(text):
    return text.split()
    
def stemmer_tokenizer(text):
    return [stemmer.stem(word) for word in tokenizer(text) if text not in stops]
    
def get_accuracy(test_labels, predictions):
    correct = 0
    for j in range(len(test_labels)):
        if test_labels[j] == predictions[j]:
            correct += 1
    return (correct/float(len(test_labels))) * 100.0

def get_accuracy_by_class(label, test_labels, predictions):
    correct = 0
    comparator = np.where(test_labels==label)[0]
    if len(comparator)>0:
        for j in range(len(comparator)):
            if test_labels[comparator[j]] == predictions[comparator[j]]:
                correct += 1
        
        return (correct/float(len(comparator))) * 100
    else:
        return 0


