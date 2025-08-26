import numpy as np
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag as treebank_pos_tag

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab') ####
nltk.download('averaged_perceptron_tagger_eng') ####
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')


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
    



class SentiWordNetProcessor:
    """Class to process sentences and extract SentiWordNet scores"""
    
    def __init__(self):
        # Cache for already processed words to speed up computation
        self.word_sentiment_cache = {}
    
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


