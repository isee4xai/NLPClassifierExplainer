import numpy as np
from nltk.stem.snowball import SnowballStemmer

from operator import itemgetter
from collections import defaultdict



### Euclidean distance and neighbour retrieval methods ###        
def get_euclidean_distance(instance1, instance2):
    '''
    Calculates euclidean distance between two instances of data
    '''
    return np.linalg.norm(np.array(instance1)-np.array(instance2))
    
def get_legacy_neighbours(training_set, test_instance, k):
    '''
    Locate most similar neighbours (i.e. neighbours with lowest euclidean distance)
    '''
    distances = []
    for x in range(len(training_set)):
        dist = get_euclidean_distance(test_instance, training_set[x])
        distances.append((training_set[x], x, dist))
    #print(distances)
    distances.sort(key = itemgetter(2))
    neighbours = []
    for x in range(k):
        neighbours.append([distances[x][0], distances[x][1], distances[x][2]])
    return neighbours    
        
### Tf-Idf word score sorting methods ###
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

### Explainability methods ###
def get_similarity_per_class(model, sources_tfidf, source_labels, query, neighbour = None):
    '''
    Method to extract the keywords for each class, as dictated by the keywords
    for each neighbour of the given query which has that class.
    Params:
        model - tfidf model to use. Must be tfidf.
        source_tfidf - the set of labelled data from which to extract neighbours (TF-IDF representtion)
        source_labels - the labels of the source texts
        query - String containing the query note.
    Returns:
        sims - Dictionary where the key is the given class label and each value 
               is the similarity of the most similar neighbour from that class to
               the query.
    '''
    
    # Transform the query using tf-idf model
    query = model.transform([query])
    
    #Identify nearest neighbours and their labels (default to five nearest neighbours)
    if neighbour == None:
        neighbour = get_legacy_neighbours(sources_tfidf.todense(), query.todense(), 5)
        
    neighbour_label = [source_labels[label[1]] for label in neighbour]
    neighbour_sim = [neigh[2] for neigh in neighbour]
    
    sims = defaultdict(list)
    for i, label in enumerate(neighbour_label):
        sims[label].append(neighbour_sim[i])
    
    for sim in sims:
        #Extract top similarity value from that class
        sims[sim] = sorted(sims[sim])[0]
        
        # Convert to similarity
        sims[sim] = (1 / (1 + sims[sim]) * 100)
    
    return sims
    
    
def get_keywords(model, query):
    '''
    Method to return a srted list of the top tf-idf values in a given query.
    Params:
        model - tfidf model to use. Must be tfidf.
        query - String containing the query note.
    Returns:
        keywords - Tuple containing a zipped list of word and tf-idf value
    '''
    # Transform the query using tf-idf model
    query = model.transform([query])
    
    # Extract vocabulary from model
    feature_names = model.get_feature_names()
    
    sorted_items = sort_coo(query.tocoo())
    keywords = extract_topn_from_vector(feature_names,sorted_items,10)

    return keywords
    
def get_keywords_per_class(model, source_tfidf, source_labels, query):
    '''
    Method to extract the keywords for each class, as dictated by the keywords
    for each neighbour of the given query which has that class.
    Params:
        model - tfidf model to use. Must be tfidf.
        source_tfidf - the set of labelled data from which to extract neighbours (TF-IDF representtion)
        source_labels - the labels of the source texts
        query - String containing the query note.
    Returns:
        keywords_per_label - Dictionary where the key is the given class label
                             and each value is a sorted list of keywords.
    '''
    
    # Transform the query using tf-idf model
    query = model.transform([query])
    
    # Extract vocabulary from model
    feature_names = model.get_feature_names()
    
    #Identify nearest neighbours and their labels
    neighbour = get_legacy_neighbours(source_tfidf.todense(), query.todense(), 5)
    neighbours = [np.asarray(neigh[0]) for neigh in neighbour]
    neighbour_label = [source_labels[label[1]] for label in neighbour]

    # Find the indice of each neighbour in the list
    labels = defaultdict(list)
    for i, label in enumerate(neighbour_label):
        labels[label].append(i)
    
    # For each label in the neighbour set
    keywords_per_label = {}
    for l in labels.keys():
        result = np.zeros(300)
        
        # For each tf-idf representation which belongs to that label
        for v in labels.get(l):
            # Sum the values
            result = np.add(result, np.asarray(neighbours[v]))
        
        # Identify the non-zero values
        word_locations = {}
        for point in range(300):
            if result[0][point] > 0:
                word_locations[point] = result[0][point]
        
        # Sort such that highest value key word is first
        word_locations = sorted(word_locations.items(), key=lambda t:t[1], reverse=True)
        
        # Extract the top n (in this case 10) key words
        sorted_items = word_locations
        keywords = extract_topn_from_vector(feature_names, sorted_items, 10)
        keywords_per_label[l] = keywords
        
    return keywords_per_label


### Tokenizing methods to divide sentences into individual words ###
def tokenizer(text):
    return text.split()
    
def stemmer_tokenizer(text):
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in tokenizer(text)]






def find_overlap(model, source_tfidf, source_labels, query, keywords = None):
    '''
    Method to extract the keywords for each class, as dictated by the keywords
    for each neighbour of the given query which has that class.
    Params:
        model - tfidf model to use. Must be tfidf.
        source_tfidf - the set of labelled data from which to extract neighbours (TF-IDF representtion)
        source_labels - the labels of the source texts
        query - String containing the query note.
    Returns:
        overlap - Dictionary where the key is the given class label
                  and each value is a 2D list containg the word (String) and
                  whether it is present in the query or not (Boolean).
    '''
    
    # Transform the query using tf-idf model
    query_words = stemmer_tokenizer(query)
    
    # If keywords are not identified, find them by calling that method
    if keywords == None:
        keywords = get_keywords_per_class(model, source_tfidf, source_labels,  query)
    
    # For each keyword
    overlap = {}
    for key in keywords:
        overlap[key] = []
        for k in keywords[key]:
            
            # Is it in the query
            if k in query_words:
                overlap[key].append([k, True])
            else:
                overlap[key].append([k, False])

    return overlap

