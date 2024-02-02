# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:52:14 2022

@author: Bruno Fleisch 
"""

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from nltk.stem.snowball import SnowballStemmer

import joblib

        
class NLPClassifier(BaseEstimator):
    
   
    def __init__(self,
                stop_words=[],
                max_features=300,
                n_neighbors=5,
                min_df = 7,
                ngram_range=(1,1)) :
        super().__init__()

        self.is_trained = False
        self.stop_words = stop_words
        self.tokenizer =  self.build_tokenizer() 
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.n_neighbors = n_neighbors
        self.stemmer = SnowballStemmer('english')

    
    def build_tokenizer(self):
        return self.stemmer_tokenizer

        
    def stemmer_tokenizer(self, text):
        tokenizer = gensim.utils.tokenize
        return   [self.stemmer.stem(word) 
            for word in tokenizer(text) if word not in self.stop_words]



    @staticmethod      
    def load_model (filename="models/trained_model.joblib"):
        '''
        
        Load a trained model from disk. 

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the model from. The default is "models/trained_model.joblib".

        Raises
        ------
        AssertionError
            If "filename" is not a valid model

        Returns
        -------
            The model.

        '''
                
        self = NLPClassifier() 
        self.pipeline = joblib.load (filename)
        self.vectorizer = self.pipeline.steps[0][1]
        self.classifier = self.pipeline.steps[1][1]
        self.is_trained = hasattr (self.classifier, "n_samples_fit_")

        return self
    
    
    def save_model (self, filename="models/trained_model.joblib"):
    
        '''
        
        Save a trained model to disk. 

        Parameters
        ----------
        filename : str, optional
            The name of the file to load the model from. The default is "models/trained_model.joblib".

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
        
        self.vectorizer = TfidfVectorizer(strip_accents=None, lowercase=False,
                        min_df=self.min_df, preprocessor=None,
                        # tokenizer=self.tokenizer,                         token_pattern=None,

                        ngram_range=self.ngram_range,
                        stop_words=self.stop_words, max_features=self.max_features)

        self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance')
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
    

    
    