from .Explanation_of_tf_idf import get_similarity_per_class, get_keywords, get_keywords_per_class, find_overlap
from wordcloud import WordCloud

class NLPClassificationExplainer:


    def __init__ (self, model):
        '''
        
        Instantiate an explainer class with specified model

        Parameters
        ----------

        model: object
            a text classification model as a sklearn pipeline that needs to have the following names steps
                vectorizer: a sklearn-compatible vectorizer (eg: tf-idf)
                classifier: a sklearn compatible classifier (eg: KNN classifier)

        '''
        if not hasattr (model, "named_steps"):
            raise "model must be a named scikit-learn pipeline"
        
        self.classifier = model.named_steps['classifier']
        self.vectorizer = model.named_steps['vectorizer']
        

    def explain (self, query, source=None, top_n=10):
        '''
        

        Parameters
        ----------
        query : str
            A note for which an recommendation has to be explained.
        
        top_n: int
            number of keywords to extract from same class neighbours 

        source: list of str
            a list of documents (strings) used to generate neighbours for explanations. 
            if source is not provided, will try to use the classifier training data if available. Otherwise,
            an exception will be thrown. 

 

        Raises
        ------
        AssertionError
        

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

        if source is not None:
            source_tfidf = self.vectorizer.transform (source)
        elif hasattr (self.classifier, '_fit_X'):
              source_tfidf  = self.classifier._fit_X
        else:
            raise AssertionError ("No source provided and classifier has no training data.")
        
        
        source_labels = self.classifier.predict (source_tfidf) 

        results = {
            "similarity_per_class": get_similarity_per_class(self.vectorizer,  source_tfidf, source_labels,  query),
            "keywords": get_keywords(self.vectorizer, query, top_n=top_n),
            "keywords_per_class": get_keywords_per_class(self.vectorizer,  source_tfidf, source_labels, query, top_n=top_n),
            "overlap": find_overlap (self.vectorizer,source_tfidf, source_labels,  query, top_n=top_n)
            }
        
        return results


    def generate_word_cloud (self, explanation, height=400, width=800,  background_color="white",
                                max_font_size=80, min_word_length=2  ):
        '''
        Generate a word cloud from an explanation. 

        Parameters
        ----------
            explanation: dict
                the dictionary returned  by the explain() method

            other parameters:
                other parameters passed to the WordCloud function

        Returns
        -------
            a WordCloud class instance

        '''



        # callback function for word cloud: 
        # returns 'green' if the word is an overalapping word,  'red' otherwise
        def _wc_color_func (word, font_size, position, orientation,
                font_path, random_state):

            overlap_1 = dict(list(explanation['overlap'].values())[0])
            if not word in overlap_1 or overlap_1[word] == False:
                return "red"
            return "green"


        wc = WordCloud(height=height, width=width, background_color=background_color,  color_func=_wc_color_func, 
                        max_font_size=max_font_size, min_word_length=2, )            
        wc.generate_from_frequencies (explanation['keywords'])
        return wc

