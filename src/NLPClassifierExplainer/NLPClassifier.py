# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:52:14 2022

@author: Bruno Fleisch 
"""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from .Explanation_of_tf_idf import get_similarity_per_class, get_keywords, get_keywords_per_class, find_overlap

from nltk.stem.snowball import SnowballStemmer
from stop_words import get_stop_words

import joblib

        
class NLPClassifier(BaseEstimator):
    
    def _tokenizer(self, text):
        return text.split()
    
    def _stemmer_tokenizer(self, text):
        return [self.stemmer_.stem(word) for word in  self._tokenizer(text) if text not in self.stops_]
    
    
    def __init__(self) :
        super().__init__()
        self.is_trained = False
        self.stemmer_ = SnowballStemmer("english")
        self.stops_   = get_stop_words('english')

        
        
    def load_model (self, filename="models/trained_model.pkl"):
        '''
        
        Load a trained model from disk. 

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the model from. The default is "models/trained_model.pkl".

        Raises
        ------
        AssertionError
            If "filename" is not a valid model

        Returns
        -------
            itself.

        '''
        
        #__main__.stemmer_tokenizer, __main__stemmer  = stemmer_tokenizer, stemmer # this really needs to be improved
        
        tmp = joblib.load (filename)
        if (tmp.__class__ != Pipeline):
            raise AssertionError ("Invalid model %s" % filename)
        
        self.pipeline   = tmp
        self.vectorizer = tmp.named_steps['tfidfvectorizer']
        self.classifier = tmp.named_steps['kneighborsclassifier']        
        self.is_trained = True
        return self
    
    
    def save_model (self, filename="models/trained_model.pkl"):
    
        '''
        
        Save a trained model to disk. 

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the model from. The default is "models/trained_model.pkl".

        Raises
        ------
        AssertionError
            If current model is not trained

        Returns
        -------
            itself.

        '''
        
        if self.is_trained is not True:
            raise AssertionError ( "No model loaded. Use load_model() method.")
            
        joblib.dump (self.pipeline, filename)
        return self
        
        
    def fit (self, X, y):
        
        self.vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False, min_df=7, preprocessor=None,
                        tokenizer=self._stemmer_tokenizer)
        self.classifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
        self.pipeline = make_pipeline( self.vectorizer,   self.classifier)
        self.pipeline.fit (X, y)
        
        self.is_trained = True
        return self
        
    
    
        
    def predict (self, X):
        '''
        
        Predict the next recommended action (class label) for the text notes X. 

        Parameters
        ----------
        X : array of str 
            An array of documents (string) representing the notes.
            
        Raises
        ------
        AssertionError
            If model is not trained.

        Returns
        -------
        Y: array of str
            Recommended next actions (class label) for each note provided. 

        '''
        
        if self.is_trained is not True:
            raise AssertionError ( "No model loaded. Use load_model() method.")
            
        return self.pipeline.predict(X)
    
    
    
    def predict_proba(self, X):
        '''
        

        Return the class probability of the given text notes. 

        Parameters
        ----------

       X : array of str 
            An array of documents (string) representing the notes. 
            
        Raises
        ------
        AssertionError
            If model is not trained.

        Returns
        -------
        Y: array of (num_samples, num_classes)
            The class probability of the input samples X. 

        '''
        
        if self.is_trained is not True:
            raise AssertionError ( "No model loaded. Use load_model() method.")
            
        return self.pipeline.predict_proba (X)
    
    
    def explain (self, query):
        '''
        

        Parameters
        ----------
        query : str
            A note for which an recommendation has to be explained. 

        Raises
        ------
        AssertionError
            If the model is not trained.

        Returns
        -------
        results : dict
            A dictionary object with the following items (dictionary keys):
                
                * similarity_per_class:  Dictionary where the key is the given class label and each value 
               is the similarity of the most similar neighbour from that class to
               the query.
                
                * keywords: Tuple containing a zipped list of word and tf-idf value
                
                * keywords_per_class: Dictionary where the key is the given class label
                             and each value is a sorted list of keywords.
                    
                * overlap:   Dictionary where the key is the given class label
                  and each value is a 2D list containg the word (String) and
                  whether it is present in the query or not (Boolean).               
        '''
        
        if self.is_trained is not True:
            raise AssertionError ( "No model loaded. Use load_model() method.")
        
            
        source_tfidf  = self.classifier._fit_X
        source_labels = self.classifier.predict (source_tfidf) 
        
        
        results = {
            "similarity_per_class": get_similarity_per_class(self.vectorizer,  source_tfidf, source_labels,  query),
            "keywords": get_keywords(self.vectorizer, query),
            "keywords_per_class": get_keywords_per_class(self.vectorizer,  source_tfidf, source_labels, query),
            "overlap": find_overlap (self.vectorizer,  self._stemmer_tokenizer, source_tfidf, source_labels,  query)
            }
        
        return results
    
    