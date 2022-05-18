# TelecomRecommendation

This task is to explain the outcomes of a NLP model that helps telecom engineers plan the different activities required for the installation of a network product on a customer premise. 

The recommendations are based on the notes provided by the network field engineers when they are sent to the customer site. These notes are processed through a NLP model that suggests the next steps they should perform or why the installation did not complete. 


## The model

the model consists of the following two components:

* a **TF-IDF vectorizer** that converts engineer notes  to a high dimensional TF-IDF representation.
* a **KNN classifier** that maps the TF-IDF converted documents to next step recommendations. 29 possible actions have been defined, represented in 29 different classes.

The model has been trained with 5,000+ notes and associated next steps.  

The Python module `TelecomNotesRecommender` can be used to access the model and generate predictions. It also has a built-in explainer that provides explanations for the predictions. 


 ## How to use the `TelecomNotesRecommender` module

See the file `example.py`  in the root folder.

### Instance creation
An model instance is created by the `Recommender` class. The trained model is loaded from disk with the `load_model()` method.  A model is provided in the repository in the `models/trained_model.pk`

```python
from TelecomNotesRecommender.Recommender import Recommender

recommender = Recommender()
recommender.load_model("models/trained_model.pk")
```
Note that the TF-IDF vectorizer and KNN classifier are available through the following two attributes:

* `vectorizer` : a scikit-learn TF-IDF vectorizer
* `classifier`: a scikit-learn KNNClassifier


### Predictions

Predictions on the next action are performed by the `predict()` method:

```python
query = "I cannot complete this task because of a hazard owned by Openreach / third party. Dp damaged by fire A1024 13416129 manager informed  is preventing further work. "
print (recommender.predict([query]))
````

output:
```['HI_']``` 

_(in this context  `HI_` is 'Hazard Indicator' and means that the heavy lifter is required.)_


### Explanations

The explanations for a given prediction are performed by the `explain()` method:

```python
explanation =  recommender.explain (query)
```

The result `explanation` is a dictionary that contains the following entries:

Key                           |   Description
---------------------|------------------------
`similarity_per_class` | Dictionary where the key is the given class label and each value is the similarity of the most similar neighbour from that class to the query.
`keywords` | Tuple containing a zipped list of word and tf-idf value
`keywords_per_class` |  Dictionary where the key is the given class label and each value is a sorted list of keywords.
`overlap` | Dictionary where the key is the given class label                   and each value is a 2D list containg the word (String) and whether it is present in the query or not (Boolean).            


                
## The task

The task of this challenge is to provide a better explainer for this type of model.  There is no specific definition of "better" in this context, we expect the participants to improve the existing solution and provide a different view point and/or different technique to generate clear, concise and precise explanations for the model. 

Targeted audience for the explanations are the field engineers who perform  the installation. They need to become confident with the model and consider it as a trusted tool to improve the quality and the efficiency of their work. For this, they need to understand how the predictions have been realized and detect any wrong/incomplete/mislabeled information that have been used by the model in the the predictions. 

The new explainer must be tested against a test dataset that is provided in this repository (in the file `test_dataset.txt` - the format is one note per line). Is is expected that the participants will perform the following

1. select 5 notes in this test dataset.
2. Run the model and predict the class of each note (the next step recommendation).
3. Provide an explanation for each prediction to ensure the objectives defined above are met.


## Submission

Participants can submit their code by creating a fork of this repository and. Once ready to submit, please submit a pull request.
       
       
## Additional Notes

### Useful attributes

You might find the following attributes of a `Recommender` instance useful:

* `vectorizer.vocabulary_` : The vocabulary learned during the training phase. It has 300 words.
* `vectorizer.idf_` : Inverse document frequency vector
* `vectorizer.stop_words_`: Terms that were ignored during the training phase
*  `classifier.classes_` : The class labels known by the classifier


### Classes definition

Class label | Definition 
------------|-----------
AER	|Aerial cable required
ARL	|ARLLAOH queue for D Pole 
NGA | (National Asset Assurance D Pole / low wire queue)  
ASA |	Asset Assurance required (Assistance Required)
C02 |	C002 - New circuit D side
C04	| C004 Plan Do Installation
C17 |	C017 D- Pole Validation	
CSQ |	CR Customer Readiness & Sales Query
CA_ |	Customer Access
DR_ |	Dig Required
DWR	|Duct work required
EER	| Exchange Equipment Required (Faulty D Side required)
FES	|Faulty E side
FWR |	Frames work required
HI_	| Hazard Indicator  (Heavy lifter required)
HSR	| Hoist required
HLR	| Hold Required
LPR	| Line Plant required
MA_ |	Manhole access
NSR |	New Site required
NA_	| No Access
NDT | 	No Dial tone required
NNR	| No NEW Activity Required
OOT	 | Out Of Time
PLR	| Planning Required
POR |	Polling Required
SR_	| Survey Required
TLR	| Track & Locate Required
TMR	 |Traffic Management required
UG_	| UG Required





