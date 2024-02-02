# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:03:55 2022

@author: Bruno Fleisch
"""

import os 
from NLPClassifierExplainer.NLPClassificationExplainer import NLPClassificationExplainer
from NLPClassifierExplainer.NLPClassifier import NLPClassifier

recommender = NLPClassifier.load_model (
    filename=os.path.join(os.path.dirname(__file__), "../models/trained_model.pkl"))

explainer = NLPClassificationExplainer (recommender)

#query = 'New dropfrom eu to carrier pole unable pu t to drop from carrier to dp as it goes over train tracks. An di cant do this safely.'
query = 'IncompleteReason: I have run out of timeithin my scheduled hours. The end customer cannot use the srvice. The work outstanding is Track.'
#query = 'IncompleteReason: Faulty fibre port ( no spares ) causing noisy line. Waiting on lums case 184241, spo ke to dcoe,expected to be resolved 5/5/18. Please pass to CS S queue Id Na. The line has been proven good to the PCP. Ear th contact was detected towards the end customer.'
#query = "IncompleteReason: a hazard owned by third party. Dp damaged by fire A1024 13416129 manager informed  is preventing further work. "

print (f"Predicted class(es): {recommender.predict([query])}")

print(f"Probability for each class: {recommender.predict_proba([query])}")

explanation = explainer.explain (query, top_n=20)
print (f"Explainer's output : {explanation}")

# generate a word cloud with the explanation

wc = explainer.generate_word_cloud(explanation)
wc.to_file(os.path.join(os.path.dirname(__file__),"../cloud.png"))



