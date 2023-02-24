from .Explanation_of_tf_idf import get_similarity_per_class, get_keywords, get_keywords_per_class, find_overlap

class NLPClassificationExplainer:


    def __init__ (self, model):
        self.classifier = model.classifier
        self.vectorizer = model.vectorizer
        

    def explain (self, query, top_n=10):
        '''
        

        Parameters
        ----------
        query : str
            A note for which an recommendation has to be explained.
            top_n - number of keywords to extract
 

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
            
        source_tfidf  = self.classifier._fit_X
        source_labels = self.classifier.predict (source_tfidf) 
        
        
        results = {
            "similarity_per_class": get_similarity_per_class(self.vectorizer,  source_tfidf, source_labels,  query),
            "keywords": get_keywords(self.vectorizer, query, top_n=top_n),
            "keywords_per_class": get_keywords_per_class(self.vectorizer,  source_tfidf, source_labels, query, top_n=top_n),
            "overlap": find_overlap (self.vectorizer,source_tfidf, source_labels,  query, top_n=top_n)
            }
        
        return results
