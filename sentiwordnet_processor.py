import numpy as np
import pandas as pd
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag as treebank_pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab') ####
nltk.download('averaged_perceptron_tagger_eng') ####
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('sentiwordnet')
nltk.download('vader_lexicon')

    
class SentiWordNetProcessor:
    ## Class to process sentences and extract SentiWordNet scores
    
    def __init__(self):
        # Cache for already processed words to speed up computation
        self.word_sentiment_cache = {}
        self.lemmatizer = WordNetLemmatizer()
    
    def get_wordnet_pos(self, treebank_tag):
        # Function to convert POS tag to a format recognized by WordNet. Source: StackOverflow.
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
    
    def get_scores(self, sentence):
        # Calculate SentiWordNet scores for a sentence
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




class SentiWordNetProcessor_NegHandling:
    
    def __init__(self):
        # Cache for already processed words to speed up computation
        self.word_sentiment_cache = {}
        self.lemmatizer = WordNetLemmatizer()
        
        # *********** NEGATION HANDLING SETUP ************
        # Common negation words that flip sentiment of a word. So if positive word preceded by these, flip the scores
        self.negation_words = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 
                              'neither', 'nowhere', 'hardly', 'scarcely', 'barely'}
        
        # Movie review specific negative words that have poor swn scores (domain-specific knowledge)
        self.movie_negative_words = {'madden', 'pointless', 'disposable', 'incoherent', 'garbage', 'annoy', 
                                    'cringeworthy', 'laughable', 'frustrate', 'waste', 'slow', 'predictable', 'forgettable', 'overhyped', 
                                    'self-indulgent', 'immature', 'irritate', 'boring', 'nonsense', 'absurd', 'amateurish', 'forget', 
                                    'generic', 'tacky', 'confuse', 'annoying', 'problematic', 'lame', 'dull', 
                                    'rubbish', 'miserable', 'nonsensical', 'pretentious', 'unwatchable', 'confusing', 'clichéd', 'disturb'}
        
        # Movie review specific positive words that have poor swn scores (domain-specific knowledge)
        self.movie_positive_words = {'fresh', 'great', 'brilliant', 'talented', 'heartwarming', 
                                    'remarkable', 'hilarious', 'superb', 'unforgettable', 'terrific', 
                                    'perfect', 'creative', 'well-crafted', 'amaze','amazing', 'breathtaking'}
        # **************************** #
    
    def get_wordnet_pos(self, treebank_tag):
        # Function to convert POS tag to a format recognized by WordNet. Source: StackOverflow.
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
    
    # *********** NEGATION DETECTION FUNCTION ************
    def is_negated(self, tokens, current_index):
        # Check if the current word is_negated or not by looking back 2 words for predefined 'negation words' 
        start_idx = max(0, current_index - 3)  # Look back up to 3 words // max because I dont want to go out of bounds
        for i in range(start_idx, current_index):
            if tokens[i].lower() in self.negation_words:
                return True
        return False
    # ************************************************* #
    
    def get_scores(self, sentence):
        # Calculate SentiWordNet scores for a sentence
        # Tokenize and tag parts of speech
        tokens = word_tokenize(sentence)
        tagged_words = treebank_pos_tag(tokens)
        tagged_words_wn = [(word, self.get_wordnet_pos(tag)) for word, tag in tagged_words]

        positive_score = 0
        negative_score = 0
        objective_score = 0
        word_count = 0

        for i, (word, pos_tag) in enumerate(tagged_words_wn):
            # Skip punctuations and words that dont carry sentiment
            if len(word) == 1 or pos_tag in ['DT', 'IN', 'TO', 'PRP', 'PRP$', 'WRB']:
                pos_tag_string = pos_tag if pos_tag is not None else "None"
                # print(f"{word:15} | POS: {pos_tag_string:4} | Skipped! ")
                continue

            # Lemmatize the word
            if pos_tag is None:
                lemma = self.lemmatizer.lemmatize(word.lower())  # Default to noun
            else:
                lemma = self.lemmatizer.lemmatize(word.lower(), pos=pos_tag)

            word = lemma
            
            # *********** SPECIAL NEGATIVE and POSITIVE WORD HANDLING ************
            # Ignore swn scores for negative words list that do not have the right scores according to me
            if word in self.movie_negative_words:
                pos_score, neg_score, obj_score = 0.1, 0.8, 0.1
                word_count += 1
                positive_score += pos_score
                negative_score += neg_score
                objective_score += obj_score

                pos_tag_string = pos_tag if pos_tag is not None else "None"
                # print(f"{word:15} | POS: {pos_tag_string:4} | Positive: {pos_score:.3f} Negative: {neg_score:.3f} Objective: {obj_score:.3f} (Domain-specific override -ve) ")
                continue  

            # Ignore swn scores for positive words list that do not have the right scores according to me
            if word in self.movie_positive_words:
                pos_score, neg_score, obj_score = 0.8, 0.1, 0.1

                if self.is_negated(tokens, i):
                    pos_score, neg_score = neg_score, pos_score  # Swap positive and negative scores 

                word_count += 1
                positive_score += pos_score
                negative_score += neg_score
                objective_score += obj_score

                pos_tag_string = pos_tag if pos_tag is not None else "None"
                # print(f"{word:15} | POS: {pos_tag_string:4} | Positive: {pos_score:.3f} Negative: {neg_score:.3f} Objective: {obj_score:.3f} (Domain-specific override +ve) ")
                continue  
            # ********************************************* #

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
            
            # *********** NEGATION HANDLING ************
            if self.is_negated(tokens, i) and (pos_score > neg_score):
                pos_score, neg_score = neg_score, pos_score  # Swap positive and negative scores
            # ******************************************
            
            pos_tag_string = pos_tag if pos_tag is not None else "None"
            # print(f"{word:15} | POS: {pos_tag_string:4} | Positive: {pos_score:.3f} Negative: {neg_score:.3f} Objective: {obj_score:.3f} ")

            positive_score += pos_score
            negative_score += neg_score
            objective_score += obj_score
            word_count += 1

        if word_count > 0:
            # Calculate averages: (note: sums would give longer sentences bigger numbers. 
            # Averaging helps to normalize scores as well as aggregate them)
            avg_positive = positive_score / word_count
            avg_negative = negative_score / word_count
            avg_objective = objective_score / word_count
            
            return avg_positive, avg_negative, avg_objective
        else:
            return 0.0, 0.0, 1.0
        



class VADERProcessor:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def get_vader_scores(self, sentence):
        # Get VADER sentiment scores for each sentence. Its a dictionary with pos, neg, neu, compound scores (4 total)
        vader_scores = self.vader_analyzer.polarity_scores(sentence)
        return vader_scores['pos'], vader_scores['neg'], vader_scores['neu'], vader_scores['compound']
    
    def get_scores(self, sentence):
        # Calculate SentiWordNet scores for a sentence
        vader_pos, vader_neg, vader_neutral, vader_compound = self.get_vader_scores(sentence)
        return vader_pos, vader_neg, vader_neutral, vader_compound        



