
import random
import numpy as np
import os 

import gensim

from sklearn.model_selection import GridSearchCV
from NLPClassifierExplainer.NLPClassifier import NLPClassifier

# ------------------------------------

SOURCES_PATH = os.path.join (os.path.dirname(__file__), "../sources")

sources = {'Aerial cable required.txt':'AER', 
           'ARLLAOH.txt':'ARL', 
           'Asset Assurance required.txt':'ASA',
           'C002 - New circuit D side.txt':'C02',
           'C004 Plan Do Installation.txt':'C04',
           'C017 D- Pole Validation.txt':'C17',
           'CR Customer Readiness & Sales Query.txt':'CSQ',
           'Complete.txt':'COM',
           'Customer Access.txt':'CA',
           'Dig Required.txt':'DR',
           'Duct work required.txt':'DWR',
           'Exchange Equipment Required.txt':'EER',
           'Faulty E side.txt':'FES',
           'Frames work required.txt':'FWR',
           'Hazard Indicator.txt':'HI',
           'Hoist required.txt':'HSR',
           'Hold Required.txt':'HLR',
           'Line Plant required.txt':'LPR',
           'Manhole access.txt':'MA',
           'New Site required.txt':'NSR',
           'No Access.txt':'NA',
           'No Dial tone required.txt':'NDT',
           #'No NEW Activity Required.txt':'NNR',
           'Out Of Time.txt':'OOT',
           'Planning Required.txt':'PLR',
           'Polling Required.txt':'POR',
           'Survey Required.txt':'SR',
           'Track & Locate Required.txt':'TLR',
           'Traffic Management required.txt':'TMR',
           'Ug Required.txt':'UG'
           }

# add path component
sources = {os.path.join (SOURCES_PATH, k):v for k,v in sources.items()}

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return(self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)



sentences = TaggedLineSentence(sources)
x = []
y = []

for sentence in sentences.to_array():
    x += [' '.join(sentence[0])]
    y += [sentence[1][0][0:3]]


x = np.asarray(x)
y = np.asarray(y)


stop_words = gensim.parsing.preprocessing.STOPWORDS

classifier = NLPClassifier()


param_grid = [{'ngram_range':[(1,1)],
               'stop_words':[list(stop_words), []],
               'max_features':[300],
               'n_neighbors': [3,5,7]}]
               
model = GridSearchCV(classifier, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
model.fit(x, y)

for idx, param in enumerate(model.cv_results_['params']):
    print (f"n_neighbors ={param['n_neighbors']}, with_stop={'yes' if param['stop_words'] else 'no'}, score={model.cv_results_['mean_test_score'][idx]}")


# display statistics about classification performance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

y_true = y_test
model.best_estimator_.fit(x_train,y_train)
y_pred = model.best_estimator_.predict (x_test)
target_names = model.best_estimator_.classifier.classes_

print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

# save best model to disk
# model.best_estimator_.save_model (filename=os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl"))

import joblib
joblib.dump (model.best_estimator_.pipeline, 
             os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl"))

