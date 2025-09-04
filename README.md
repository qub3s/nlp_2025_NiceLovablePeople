# DNLP SS25 Final Project 

<div align="left">

<b> NiceLoveablePeople </b> <br/>

Esther Hagenkort: esthako/GOESTERN-1006113 <br/>

Georg Eckardt: qub3s <br/>

Hamza Ahmed Siddiqui: hamzasiddiqui10 <br/>

Amon Pönitzsch: 4m0n <br/>

Leonardo Christian da Camara Silva: Dacasil <br/>

</div>

  
## Introduction
This repository is our baseline implementation of the project for the model M.Inf.2202: Deep Learning for Natural Language Processing at the University of Göttingen by the GippLab.

Aditionally to completing the bert implementation and the optimizer, the following tasks have been completed:
- Stanford Sentiment Treebank (SST) - Sentiment analysis
- Quora Question Pairs (QQP) - Question similarity
- Semantic Textual Similarity (STS) - Measuring text meaning similarity
- Paraphrase Type Detection (PTD) - Identifying paraphrase types and relationships
- Paraphrase Type Generation (PTG) - Generating diverse paraphrase types
- Bonus: Paraphrase Type Detection with Bert (PTD-Bert) - Identifying paraphrase types and relationships

## Implementation & Contribution
We followed the instructions and adapted hyperparameters when needed to avoid overfitting.

Esther Hagenkort: 
- bert.py (revise, comment)
- bonus: paraphrase type detection with bert (PTD-bert) (debugging)
- Paraphrase Type Generation (PTG)

Georg Eckardt: 
- optimizer
- bert.py
- paraphrase type detection

Hamza Ahmed Siddiqui:
- Group name
- bert.py (revise, comment)
- Stanford Sentiment Treebank (SST) - Sentiment analysis

Amon Pönitzsch:
- paraphrase detection (QQP)

Leonardo Christian da Camara Silva:
- Semantic Textual Similarity (STS)
- bonus: paraphrase type detection with bert (PTD-bert)
  


## Results Part 1
For the baselines we reached the following results with the described hyperparameters and metrics.
We noticed that the results changed from run to run even if the hyperparameters were not changed. We therefore assume that something with the seed is not working.

### Stanford Sentiment Treebank (SST) - Sentiment analysis

**Hyperparameters:**
- mode: `finetune`

- epochs: `6`

- learning rate: `1e-05`

- optimizer: `AdamW`

- dropout rate: `0.25`

- batch size: `64`

**Training results:**
- Accuracies Epoch 6: `train :: 0.896`, `dev :: 0.521`


### Quora Question Pairs (QQP) - Question similarity
**Hyperparameters:**
- mode: `finetune`

- epochs: `1`

- learning rate: `8e-5`

- optimizer: `AdamW`

- dropout rate: `0.1`

- batch size: `64`


**Training results:**
- Dev Accuracy Finetuning: `0.870`

### Semantic Textual Similarity (STS) - Measuring text meaning similarity
**Hyperparameters:**
- mode: `finetune`

- epochs: `10`

- learning rate: `1e-5`

- optimizer: `AdamW`

- dropout rate: `0.25`

- batch size: `64`

**Training results:**
- Dev Correlation STS Finetuning: `0.371`

### Paraphrase Type Detection (PTD) - Identifying paraphrase types and relationships
**Hyperparameters:**
- epochs: `10`

- learning rate: `1e-5`

- optimizer: `AdamW`

- dropout rate: `0.1`

- batch size: `64`


**Training results:**
- Paraphrase Detection: Dev Accuracy: `0.906`
### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types

**Hyperparameters:**
- epochs: `5`

- learning rate: `1e-5`

- eps = `1e-8`

- optimizer: `AdamW`

- batch size: `32`
**Training results:**
- Paraphrase Generation: BLEU Score: `49.18`
  
### Bonus: Paraphrase Type Detection with Bert (PTD-bert) - Identifying paraphrase types and relationships

**Hyperparameters:**
- epochs: `10`

- learning rate: `1e-6`

- eps = `1e-8`

- optimizer: `AdamW`

- batch size: `16`
  
**Training results:**
  - Dev Accuracy `0.13`
  - Dev Macro f1 `0.264`
  - Dev Micro f1 `0.709`

**Evaluation structure:**
The ETPC train label distribution (mean label value per class): 
[0.138, 0.049, 0.048, 0.127, 0.179, 0.647, 0.109, 0.049, 0.001, 0.005, 0.198, 0.013, 0.041, 0.006, 0.017, 0.011, 0.112, 0.192, 0.018, 0.075, 0.764, 0.207, 0.092, 0.993, 0.164, 0.023]

It indicates that most classes are very rare (mean values close to 0.01-0.05) and a few classes are very common (e.g. class 23 with 0.993). So it follows that we have severe class imbalance, so its hard to learn rare classes and therefore the standard accuracy (fraction of correct labels) might be missleading. Thats why I decided to also display the macro and micro F1.

**Improvements:**
For the second part I will do hyperparameter finetuning to counter overfitting and find the best local minima. Also I will try to impliment the "Siamese + interaction" recipe used by DeBErta on ETPC from the GippLab group (Paraphrase Types for Generation and Detection, Wahle et al.) and other improvements from Chapter 7.






## Results Part 2
### Stanford Sentiment Treebank (SST) - Sentiment Analysis
The vanilla implementation of the sentiment prediction with minBERT gave us a baseline of ~52% dev set accuracy. 

To improve the model a myriad of research questions were posited and then worked on. Some attempted improvements yielded increases in the dev accuracy, some did not affect the accuracy although they were sensible, and some worsened the dev accuracy. All of these are summarized below:


**1. Is attention masking in the self-attention layer for a sentiment classification task hurting performance?**

**Explanation:** In our Part-01 submission we had to complete bert.py according to documentation so we had implemented the attention masking so that model only has left-context. For a sentiment classification task, bi-directional context will always be available and will always be more powerful.

**Experiment:** I commented out the attention masking in `bert.py`. The results did not improve significantly and in my understanding, its because the model needs to be pretrained with bidirectional context properly for this alteration to impact results.

**2. Can we solve the clear overfitting problem in the baseline model?**

**Explanation:** In baseline model’s training the training accuracy increases consistently and reaches 90%+ whereas dev accuracy plateaued around 51% and even starts going down in later epochs. This shows directly that the baseline model is suffering from overfitting on the training data. The solution can be some hyperparameter tuning.

**Experiment:** 
- L2 regularization: in the baseline implementation we did not have any regularization so the weights could grow indefinitely. After looking at documentation for AdamW, I found that the weight_decay parameter can enable L2 regularization. I tried 0.01 and 0.025 and settled on the latter. 

- Learning Rate: a default learning rate of 1e-5 almost always lead to quick convergence of training accuracy to 90%+ values. I experimented with lower values and found that 1e-6 is the ideal value as the dev accuracy increases gradually and while keeping the train accuracy under control so the model will generalize much better. I had to increase the epochs from 10 to 20 which made training all latter models more expensive but it was worth it because the models will now generalize much better.

- Result: The results showed clearly that the train accuracy was now more in tandem with the dev accuracy but there was no discernible improvement in the baseline dev accruacy.


**3. Can document-level sentiment scores aid minBERT in sentiment classification?**

**Explanation:** Our work is inspired by the feature fusion approach of Hoover et al. (2020), but we implement a simplified, document-level variant. Instead of performing complex phrase-level composition, we calculate aggregate SentiWordNet scores for the entire input sentence and concatenate them directly with the [CLS] token embedding before classification. This provides the model with a strong prior sentiment signal without additional computational overhead.

The baseline model uses the final 768-d CLS token’s hidden state to predict sentiment of a movie review. The idea was whether we can enrich this input to the classification head by using a lexical resource like WordNet wherein alongside the 768 dimensions of CLS token which encode “sentence sentiment through self-attention mechanism”, we also get some sentiment scores to the classification head?
I used SentiWordNet (SWN) which assigns sentiment scores to WordNet synsets. Each synset gets three scores:
- Positive score (0.0 to 1.0)
- Negative score (0.0 to 1.0)
- Objective score (0.0 to 1.0) The objective score tells us how neutral or factual a word is. A higher score means the word carries less emotional sentiment.

The 3 scores always sum to 1.0 for each synset.
These scores enrich the input space by telling the model the overall positivity or negativity of a sentence through a statistical approach. Keep in mind these scores completely ignore context and are word level features aggregated down to document-level.

Note: I used averaging for aggregation over summing because the latter would have given much more weight to longer sentences.

Citation: SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics:
https://aclanthology.org/2020.acl-main.341.pdf


**Experiment:** To implement this I had to break down the problem in multiple sub-problems which were solved as follows:

i. Analyzed `datasets.py` to find how to access the raw 'sentence' column of csv for each batch of the dataloader easily. Found that we can use `batch['sents']`. This is crucial because my changes can work on any seen/unseen dataset now.

ii. Wrote `sentiwordnet_processor.py` module which downloads NLTK dependencies and  `SentiWordNetProcessor` class which takes in a sentence, calculates scores for each word in the sentence and aggregates the scores to get sentence-level positive score and negative score. Lemmatization before scoring was added to enable score calculation for more words per sentence. I also get the avg_obj_score feature here but avoided using it as a feature because it added noise to the model and was hurting performance of my classifier.

iii. Altered the `predict_sentiment()` function of `MultitaskBERT` class so it can calculate SWM’s avg_pos_score and avg_neg_score and then append them in the CLS token’s vector before classification.

iv. Slight changes in the training loop to make sure that the raw sentences are being accessed for each batch and passed into the `predict_sentiment()` function. (`evaluation.py` changes same as training loop)

v. Installing NLTK on cluster: `setup_gwdg_nltk.sh`

Results: Dev accuracy improved to 0.55 and became more stable during fine tuning so this was a very good addition.

**4. Does the classification head architecture require complexity?**

**Explanation:** The CLS token is mapped onto the classification head directly after concatenation of SWN scores. I thought maybe adding some layers in between will help learn even more complex relationships between 768 features from BERT and the additional SWN scores.

**Experiment:** I tried two different architectures:
- 770 → 128 → 16 → 5
- 770 → 64 → 5

But both performed poorly by reducing dev accuracy (0.49) so reverted back to simple 770 → 5.

**5. Can we engineer new features from the SentiWordNet positive and negative scores to improve classification?**

**Explanation:** Engineer 3 new features from pos_score and neg_score (strength, ratio, net)

**Experiment:**
- sentiment_strength = avg_pos_score + avg_neg_score  # How strong the sentiment is
- sentiment_ratio = avg_pos_score / (avg_neg_score + 1e-8) if avg_neg_score > 0 else 10  # Pos/Neg ratio
- net_sentiment = avg_pos_score - avg_neg_score  # Net sentiment score
           
Results stayed similar. No discernible improvements but still a decent addition to keep.


**6. Is the SentiWordNetProcessor performing similarly on all classes?**

**Explanation:** Error analysis revealed a significant performance disparity, with poorer accuracy on negative reviews (classes 0-1) compared to positive ones (classes 3-4). We hypothesized that this was due to the prevalence of negated positive statements (e.g., 'not good') in negative reviews, which our initial SentiWordNet processor misinterpreted as positive. To address this, we implemented a negation handling module inspired by the classic technique of Turney (2002). This module flips the positive and negative sentiment scores of a word if it is preceded by a negation term (e.g., 'not', 'no', 'never'), thereby correctly interpreting phrases like 'not good' as negative.

Citation:
Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews:
https://aclanthology.org/P02-1053.pdf 

**Experiment:** “Negation handling” functionality was added to the processor in and wrote a new class `SentiWordNetProcessor_NegHandling` which flips positive and negative scores of a word if the word is preceded by a negating word like “no”, “not”, “barely” etc.
~~~
I looked at the problem with more granularity and realized some domain-specific intervention in the wordnet sentiment scores is required for the lexical database to perform better on specifically movie review problems., I asked DeepSeek (AI) to give me a of movie-reviews domain’s:
List of common positive sentiment words list
List of common negative sentiment words list

The original list of words did not come from the SST data so there is no risk of introducing bias.

I found some words (both negative and positive) whose SWN scores I over-ride with my custom scores. Lists of these words can be found in the `SentiWordNetProcessor_NegHandling` class. 

 
In the end I settled on applying negation handling only on positive words because that showed the best results.
~~~




**7. Should we trust the minBERT model or the SWN scores equally?**

**Explanation:** The model has two distinct components that it uses to predict sentiment of a sentence:
1. min BERT h_cls token’s 768 features
2. SWN’s 5 features

The weight of these two sources is equal at the moment but in reality one source must be better or worse than the other. How much do I trust each source is the million dollar question. My idea is to use a learned gating system to let the model dynamically decide how much to trust SWN vs BERT features for sentiment classification.

**Experiment:** Instead of just concatenating SWN scores to h_cls, I introduced a simple learned gating mechanism to let the model dynamically decide how much to trust SWN vs BERT features for each example.

It's a simple NN which takes in h_cls+SWN scores and outputs two weights (0.0 < weight < 1.0) using a sigmoid function. First weight for BERT features and second weight for the SWN scores.

Architecture: (768+5) → 256 → 2

The bert_weight is multiplied element-wise to 768 BERT features, and the swn_weight is multiplied element-wise to the 5 SWN features before all are concatenated and mapped to the classification head.

Lastly, instead of a sigmoid function, a softmax was also tried to make the weights sum up to 1.0 but it reduced dev accuracy so sigmoid was chosen at the end.




**Summary:**
| Sno.| Experiment | Dev Accuracy |
|---|--------------|--------------|
| 0 | Baseline | 0.519 |
| 1 | Remove attention masking from BERT | 0.519 |
| 2 | Hyperparameter tuning to solve overfitting | 0.52 |
| 3 | SWN score (positive and negative) added to h_cls | 0.55 |
| 4 | Add dense layers between h_cls and classification head| 0.49 (reverted) |
| 5 | Engineer 3 new features from SWN positive and negative scores| 0.54 |
| 6 | Negation Handling and domain-specific knowledge added to SWN processor | 0.54 |
| 7 | Gating mechanism added to weight BERT model and SWN scores| 0.54 |







-------------------------



### Grete Cluster
To run the tasks on the Grete cluster we adapted and used the `run_train.sh` script given to us.

## AI-Usage 
AI (debugging) support such as ChatGPT were used, a detailed AI-Usage card will be provided in the final report.

## Acknowledgement
The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.
