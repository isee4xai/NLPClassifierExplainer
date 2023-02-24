# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:03:55 2022

@author: Bruno Fleisch
"""

import os 
from NLPClassifierExplainer.NLPClassificationExplainer import NLPClassificationExplainer
from NLPClassifierExplainer.NLPClassifier import NLPClassifier

recommender = NLPClassifier.load_model (
    filename=os.path.join(os.path.dirname(__file__), "../models/trained_model.joblib"))

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

from wordcloud import WordCloud


# callback function for word cloud: 
# returns 'green' if the word is an overalapping word,  'red' otherwise
def wc_color_func (word, font_size, position, orientation,
        font_path, random_state):

    overlap_1 = dict(list(explanation['overlap'].values())[0])
    if not word in overlap_1 or overlap_1[word] == False:
        return "red"
    return "green"


wc = WordCloud(height=400, width=800, background_color="white",  color_func=wc_color_func,
max_font_size=80, min_word_length=2, )
wc.generate_from_frequencies (explanation['keywords'])
wc.to_file(os.path.join(os.path.dirname(__file__),"../cloud.png"))



