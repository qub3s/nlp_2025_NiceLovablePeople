import numpy as np
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag as treebank_pos_tag
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab') ####
nltk.download('averaged_perceptron_tagger_eng') ####
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')

    
class SentiWordNetProcessor:
    """Class to process sentences and extract SentiWordNet scores"""
    
    def __init__(self):
        # Cache for already processed words to speed up computation
        self.word_sentiment_cache = {}
        self.lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, treebank_tag):
        """Function to convert POS tag to a format recognized by WordNet. Source: StackOverflow."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None #treebank_tag
    
    def get_swn_scores(self, sentence):
        """Calculate SentiWordNet scores for a sentence"""
        # Tokenize and tag parts of speech
        tokens = word_tokenize(sentence)
        tagged_words = treebank_pos_tag(tokens)
        tagged_words_wn = [(word, self.get_wordnet_pos(tag)) for word, tag in tagged_words]

        positive_score = 0
        negative_score = 0
        objective_score = 0
        word_count = 0

        for word, pos_tag in tagged_words_wn:
            # Skip punctuations and words that dont carry sentiment
            if len(word) == 1 or pos_tag in ['DT', 'IN', 'TO', 'PRP', 'PRP$', 'WRB']:
                continue

            # Lemmatize the word
            if pos_tag is None:
                lemma = self.lemmatizer.lemmatize(word.lower())  # Default to noun
            else:
                lemma = self.lemmatizer.lemmatize(word.lower(), pos=pos_tag)

            word = lemma

            # Check cache first
            cache_key = f"{word}_{pos_tag}"
            if cache_key in self.word_sentiment_cache:
                pos_score, neg_score, obj_score = self.word_sentiment_cache[cache_key]
            else:
                # Get synsets for the word
                synsets = wn.synsets(word, pos=pos_tag)
                if not synsets:
                    pos_score, neg_score, obj_score = 0.0, 0.0, 1.0
                else:
                    # Use the first synset - Most Frequent Sense (MFS)
                    synset = synsets[0]
                    try:
                        swn_synset = swn.senti_synset(synset.name())
                        pos_score = swn_synset.pos_score()
                        neg_score = swn_synset.neg_score()
                        obj_score = swn_synset.obj_score()
                    except:
                        pos_score, neg_score, obj_score = 0.0, 0.0, 1.0
                
                # Add to cache
                self.word_sentiment_cache[cache_key] = (pos_score, neg_score, obj_score)
            
            positive_score += pos_score
            negative_score += neg_score
            objective_score += obj_score
            word_count += 1

        if word_count > 0:
            # Calculate averages
            avg_positive = positive_score / word_count
            avg_negative = negative_score / word_count
            avg_objective = objective_score / word_count
            
            return avg_positive, avg_negative, avg_objective
        else:
            return 0.0, 0.0, 1.0








class SentiWordNetProcessor_neg_handling:
    """Class to process sentences and extract SentiWordNet scores"""
    
    def __init__(self):
        # Cache for already processed words to speed up computation
        self.word_sentiment_cache = {}
        # ***** Negation words dictionary
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 
                              'neither', 'nowhere', 'hardly', 'scarcely', 'barely'}
        # ***** Intensifier words with multipliers
        self.intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'absolutely': 2.0, 'completely': 1.8,
            'totally': 1.8, 'utterly': 2.0, 'highly': 1.5, 'really': 1.3,
            'quite': 1.2, 'somewhat': 0.8, 'slightly': 0.7, 'barely': 0.5
        }
    
    def get_wordnet_pos(self, treebank_tag):
        """Function to convert POS tag to a format recognized by WordNet. Source: StackOverflow."""
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    
    def detect_negation_context(self, tokens, current_index):
        """ ***** Check if current word is in negation context """
        # Look back 2-3 words for negation
        start = max(0, current_index - 3)
        for i in range(start, current_index):
            if tokens[i].lower() in self.negation_words:
                return True
        return False
    
    def get_intensifier_strength(self, tokens, current_index):
        """ ***** Check if current word is intensified """
        # Look back 1-2 words for intensifiers
        start = max(0, current_index - 2)
        for i in range(start, current_index):
            if tokens[i].lower() in self.intensifiers:
                return self.intensifiers[tokens[i].lower()]
        return 1.0  # Default multiplier
    
    def get_swn_scores(self, sentence):
        """Calculate SentiWordNet scores for a sentence"""
        # Tokenize and tag parts of speech
        tokens = word_tokenize(sentence)
        tagged_words = treebank_pos_tag(tokens)
        tagged_words_wn = [(word, self.get_wordnet_pos(tag)) for word, tag in tagged_words]

        positive_score = 0
        negative_score = 0
        objective_score = 0
        word_count = 0

        # ***** Movie review specific negative patterns
        movie_negative_patterns = {
            'waste', 'boring', 'terrible', 'awful', 'horrible', 'disappointing',
            'bad', 'poor', 'weak', 'stupid', 'ridiculous', 'pointless', 'mess'
        }
        
        for i, (word, pos_tag) in enumerate(tagged_words_wn):
            # Skip punctuations and words that dont carry sentiment
            if len(word) == 1 or pos_tag in ['DT', 'IN', 'TO', 'PRP', 'PRP$', 'WRB']:
                continue
            
            # ***** Check for movie-specific negative words (override SentiWordNet)
            if word.lower() in movie_negative_patterns:
                positive_score += 0.0
                negative_score += 0.8  # Strong negative bias for these words
                objective_score += 0.2
                word_count += 1
                continue
                
            # Check cache first
            cache_key = f"{word}_{pos_tag}"
            if cache_key in self.word_sentiment_cache:
                pos_score, neg_score, obj_score = self.word_sentiment_cache[cache_key]
            else:
                # Get synsets for the word
                synsets = wn.synsets(word, pos=pos_tag)
                if not synsets:
                    pos_score, neg_score, obj_score = 0.0, 0.0, 1.0
                else:
                    # Use the first synset - Most Frequent Sense (MFS)
                    synset = synsets[0]
                    try:
                        swn_synset = swn.senti_synset(synset.name())
                        pos_score = swn_synset.pos_score()
                        neg_score = swn_synset.neg_score()
                        obj_score = swn_synset.obj_score()
                    except:
                        pos_score, neg_score, obj_score = 0.0, 0.0, 1.0
                
                # Add to cache
                self.word_sentiment_cache[cache_key] = (pos_score, neg_score, obj_score)
            
            # ***** Apply negation handling
            if self.detect_negation_context(tokens, i):
                # Flip sentiment for negated words
                pos_score, neg_score = neg_score, pos_score
            
            # ***** Apply intensifier multiplier
            intensity = self.get_intensifier_strength(tokens, i)
            pos_score *= intensity
            neg_score *= intensity
            
            positive_score += pos_score
            negative_score += neg_score
            objective_score += obj_score
            word_count += 1

        if word_count > 0:
            # Calculate averages
            avg_positive = positive_score / word_count
            avg_negative = negative_score / word_count
            avg_objective = objective_score / word_count
            
            # ***** Additional bias for strongly negative sentences
            if avg_negative > 0.4 and avg_positive < 0.1:
                avg_negative = min(1.0, avg_negative * 1.2)  # Boost negative score
            
            return avg_positive, avg_negative, avg_objective
        else:
            return 0.0, 0.0, 1.0







## development work ##

def get_wordnet_pos(treebank_tag):
    """Function to convert POS tag to a format recognized by WordNet. Source: StackOverflow."""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN #treebank_tag
    

def get_swn_scores(sentence): 
    # Tokenize and tag parts of speech
    tokens = word_tokenize(sentence)
    tagged_words = treebank_pos_tag(tokens)
    tagged_words_wn = [(word, get_wordnet_pos(tag)) for word, tag in tagged_words]

    positive_score = 0
    negative_score = 0
    objective_score = 0
    word_count = 0

    # print(f"Sentence: '{sentence}'")
    # print(f"WN Tagged Tokens: '{tagged_words_wn}'")

    for word, pos_tag in tagged_words_wn:

        # Skip punctuations and words that dont carry sentiment
        if len(word) == 1 or pos_tag in ['DT', 'IN', 'TO', 'PRP', 'PRP$', 'WRB']:
            print(f"Word: {word:12} |")
            continue
            
        # Get synsets for the word
        synsets = wn.synsets(word, pos=pos_tag)
        if not synsets:
            continue
            
        # Use the first synset - Most Frequent Sense (MFS)
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        
        # Accumulate scores
        positive_score += swn_synset.pos_score()
        negative_score += swn_synset.neg_score()
        objective_score += swn_synset.obj_score()
        word_count += 1
        
        # Print word analysis
        # print(f"Word: {word:12} | POS: {pos_tag:4} | Positive: {swn_synset.pos_score():.3f} | "
        #         f"Negative: {swn_synset.neg_score():.3f} | Objective: {swn_synset.obj_score():.3f}")


    if word_count > 0:
        # Calculate averages
        avg_positive = positive_score / word_count
        avg_negative = negative_score / word_count
        avg_objective = objective_score / word_count
        
        return avg_positive, avg_negative, avg_objective
    else:
        return 0.0, 0.0, 1.0