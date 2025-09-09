# DNLP SS25 Final Project 

<h3>Group Name: NiceLovablePeople</h3>

Group code: G04

Group repository:

Tutor responsible: Niklas Bauer

Group team leader: Esther Hagenkort: esthako/GOESTERN-1006113 

Group members:
- Georg Eckardt: qub3s
- Hamza Ahmed Siddiqui: hamzasiddiqui10
- Amon Pönitzsch: 4m0n
- Leonardo Christian da Camara Silva: Dacasil


## Introduction
This repository is our implementation of the project for the model M.Inf.2202: Deep Learning for Natural Language Processing at the University of Göttingen by the GippLab.

Adsitionally to completing the bert implementation and the optimizer, the following tasks have been completed:
- Stanford Sentiment Treebank (SST) - Sentiment analysis
- Quora Question Pairs (QQP) - Question similarity
- Semantic Textual Similarity (STS) - Measuring text meaning similarity
- Paraphrase Type Detection (PTD) - Identifying paraphrase types and relationships
- Paraphrase Type Generation (PTG) - Generating diverse paraphrase types
- Bonus: Paraphrase Type Detection with Bert (PTD-Bert) - Identifying paraphrase types and relationships

## Contributions
We followed the instructions and adapted hyperparameters when needed to avoid overfitting.

Esther Hagenkort: 
- bert.py (revise, comment)
- bonus: paraphrase type detection with bert (PTD-bert) (debugging)
- Paraphrase Type Generation (PTG)

Georg Eckardt: 
- optimizer
- bert.py
- Paraphrase Type Detection - PTD

Hamza Ahmed Siddiqui:
- Came up with group name 
- bert.py (revise, comment)
- Stanford Sentiment Treebank (SST) - Sentiment analysis

Amon Pönitzsch:
- paraphrase detection (QQP)

Leonardo Christian da Camara Silva:
- Semantic Textual Similarity (STS)
- bonus: paraphrase type detection with bert (PTD-bert)

Georg Eckardt: qub3s <br/>

## Methodology


## Experiments and Results

### Stanford Sentiment Treebank (SST) - Sentiment Analysis
The vanilla implementation of the sentiment prediction with minBERT gave us a baseline of 0.521 dev accuracy in Part-01. 

To improve the model, a myriad of research questions were posited and then worked on. The main target to measure model's performance was dev set accuracy after each experiment. Some experiments yielded increases in the dev accuracy, some did not affect the accuracy although they were sensible, and some worsened the dev accuracy. In this section, all of the major improvement attempts are explained in order and at the end you will find a summary table which will show you dev accuracy after each change was implemented.

The state-of-the-art model for the SST dataset has ~59.8% dev accuracy so room for improvement from baseline is slim. And even 0.5% improvements in accuracy are significant for this problem.

<h3> 1. Is attention masking in the the self-attention layer for a sentiment classification task hurting performance? </h3>
<details> 

**Explanation:** In our Part-01 submission we had to complete bert.py according to documentation so we had implemented the attention masking so subsequently the model only has left-context. For a sentiment classification task, bi-directional context will always be available and will always be more powerful so my expectation is a slight improvement in dev accuracy.

**Implementation:** Code change was straightforward whereby I had to comment out the attention-masking in `bert.py`. 
```
X = X + attention_mask ## HS: attention mask removed for bidirectional context
```
The results did not improve and my understanding is that the model needed to be pretrained with bi-directional context properly for this alteration to impact results significantly. Since we are only finetuning the model the impact on results is negligible.

</details>


<h3>2. Can we solve the clear overfitting problem in the baseline model?</h3>
<details> 

**Explanation:** In baseline model’s training, the training accuracy increases consistently and reaches 90%+ whereas dev accuracy plateaued around 52% and even starts going down in later epochs. This shows directly that the baseline model is suffering from overfitting on the training data. The solution can be some hyperparameter tuning.

**Implementation:** 
- L2 regularization: in the baseline implementation we did not have any regularization so the weights could grow indefinitely. After looking at documentation for AdamW, I found that the weight_decay parameter can enable L2 regularization. I tried 0.01 and 0.025 and settled on the latter. 

- Learning Rate: a default learning rate of 1e-5 almost always lead to quick convergence of training accuracy to 90%+ values. I experimented with lower values and found that 1e-6 is the ideal value as the dev accuracy increases gradually and while keeping the train accuracy under control so the model will generalize much better. 

The expectation was that with L2 regularization and the smaller learning rate, the model will converge slowly so I had to increase the epochs from 10 to 25. 

**Results:** The results showed clearly that the train accuracy was now more in tandem with the dev accuracy but there was no discernible improvement in the baseline dev accruacy. These changes made training models more much more expensive but it is worth it because the models will now generalize much better. (Note: Although solving overfitting is a positive change, I reverted to original (lr=1e-5 / epochs=10) settings for the remainder of my experiments for quicker trainings and validations). 

Additional idea that was not implemented: Introduce an early-stopping mechanism wherein the training stops if improvement in the dev accuracy in an epoch relative to 3 previous epochs is below a threshold. But training for this problem with baseline hyperparameters was fairly quick so this idea was not pursued further.

</details> 

<h3>3. Can document-level sentiment scores from a lexical database aid minBERT in sentiment classification?</h3>

<details> 

**Explanation:** This work is inspired by the feature fusion approach of Hoover et al. (2020), but I will implement a simplified, document-level variant. Instead of performing complex phrase-level composition, we calculate aggregate SentiWordNet scores for the entire input sentence and concatenate them directly with the h_cls embedding before classification. This provides the model with a strong prior sentiment signal without additional computational overhead.

The baseline model uses the final 768-d CLS token’s hidden state to predict sentiment of a movie review. The idea was whether we can enrich this input to the classification head by using a lexical resource like WordNet wherein alongside the 768 features of the CLS token which encode “sentence sentiment through self-attention mechanism”, we also get some sentiment scores to the classification head.

I used SentiWordNet (SWN) which assigns sentiment scores to WordNet synsets. Each synset has three SWN scores:
- Positive score (0.0 to 1.0)
- Negative score (0.0 to 1.0)
- Objective score (0.0 to 1.0) The objective score tells us how neutral or factual a word is. A higher score means the word carries less emotional sentiment.

The 3 scores always sum to 1.0 for each synset.

These scores enrich the input space by telling the model the overall positivity or negativity of a sentence through a statistical approach. Keep in mind these scores completely ignore context and are word level features aggregated down to document-level. (Note: I used averaging for aggregation over summing because the latter would have given much more weight to longer sentences.)

I initially used only positive and negative scores in my experiment. Adding objective score to the feature space reduced the dev accuracy slightly so ultimately decided to not use the objective score. It is unclear what the problem was because multi collinearity is not a problem for deep learning that is why usually more features are better and the network learns to ignore useless features. But the objective score feature was adding some noise to the model hurting accuracy so it was removed.

Paper: SentiBERT: A Transferable Transformer-Based Architecture for Compositional Sentiment Semantics
https://aclanthology.org/2020.acl-main.341.pdf


**Implementation:** To implement this I had to break down the problem in multiple sub-problems which were solved as follows:

1. Analyzed `datasets.py` to find how to access the raw 'sentence' column of csv for each batch of the dataloader easily. Found that we can use `batch['sents']`. This is crucial because I wanted to develop a system wherein I can get SWN scores for any seen/unseen datasets.

2. Wrote `sentiwordnet_processor.py` module which downloads NLTK dependencies and the `SentiWordNetProcessor` class which takes in a sentence, calculates scores for each word in the sentence and aggregates the scores to get sentence-level positive score and negative score. Lemmatization before scoring was added to enable score calculation for more words per sentence. Most Frequent Sense (MFS) synset was used for every word to get the sentiment scores.

3. Altered the `predict_sentiment()` function of `MultitaskBERT` class so it can calculate SWM’s avg_pos_score and avg_neg_score and then append them in the CLS token’s vector before the classification head.

4. Slight changes in the training loop to make sure that the raw sentences are being accessed for each batch and passed into the `predict_sentiment()` function. (`evaluation.py` changes same as training loop)

5. Installing NLTK on cluster: `setup_gwdg_nltk.sh`

The expectation was that this ensemble approach containing both BERT and SentiWordNet information for predicting sentiment will have a significant improvement in dev accuracy.

**Results:** Dev accuracy improved to 0.532 and became more stable during fine tuning so this met my expectations and turned out to be a very good change.

</details> 

<h3>4. Does the classification head architecture require complexity?</h3>

<details> 

**Explanation:** The CLS token is mapped onto the classification head directly after concatenation of SWN scores. I thought maybe adding some layers in between will help learn even more complex relationships between 768 features from BERT and the additional SWN scores.

**Implementation:** I tried two different architectures to go from 770 features to 5 classes in the last layer:
- (768+2) → 128 → 16 → 5
- (768+2) → 64 → 5

**Results:** Both architectures performed poorly by reducing dev accuracy to around 0.49 so reverted back to simple 770 → 5 architecture.

<h3>5. Can we engineer new features from the SentiWordNet positive and negative scores to improve classification?</h3>

**Explanation:** Engineer 3 new features from pos_score and neg_score (strength, ratio, net)

**Implementation:**
- sentiment_strength = avg_pos_score + avg_neg_score  # How strong the sentiment is
- sentiment_ratio = avg_pos_score / (avg_neg_score + 1e-8) if avg_neg_score > 0 else 10  # Pos/Neg ratio
- net_sentiment = avg_pos_score - avg_neg_score  # Net sentiment score

My expectations were this will result in slight improvement, not major because the information captured inside the raw pos_score and neg_score was already represented, these new features merely describe some interactions between the raw features. But my previous attempt at capturing said interactions between raw features i.e. experiment #4 failed so I was skeptical.

**Results:** Dev accuracy stayed very similar with no noteworthy increase but I still believe this is a reasonable addition to keep in future models. Intuitively these are good features and enrich the raw SWN feature space.

</details> 

<h3>6. Is the SentiWordNetProcessor performing similarly on all classes?</h3>

<details> 

**Explanation:** Error analysis revealed a significant performance disparity, with poorer accuracy on negative reviews (classes 0-1) compared to positive ones (classes 3-4). We hypothesized that this was due to the prevalence of negated positive statements (e.g., 'this movie is "not good"') in negative reviews, which our initial SentiWordNet processor misinterpreted as positive. To address this, we implemented a negation handling module inspired by the classic technique of Turney (2002). This module flips the positive and negative sentiment scores of a word if it is preceded by a negation term (e.g., 'not', 'no', 'never'), thereby correctly interpreting phrases like 'not good' as negative.

Paper: Thumbs Up or Thumbs Down? Semantic Orientation Applied to Unsupervised Classification of Reviews 
https://aclanthology.org/P02-1053.pdf 

**Implementation:** “Negation handling” functionality was added to the swn processor and wrote a new class `SentiWordNetProcessor_NegHandling` which flips positive and negative scores of a word if the word is preceded by a negating word like “no”, “not”, “barely” etc. To determine if a given word is negated, I looked back 3 words in the text for a negating word.

My experimentation showed that negating (flipping) only on positive scores yielded slightly better results so in the finalized version I am only doing negation handling for words deemed as positive by SWN.

Error analysis also allowed me to look at the problem with more granularity and I realized some domain-specific intervention in the wordnet sentiment scores is required for the lexical database to perform better on the specific domain of movie reviews. I asked DeepSeek (AI) to give me:
- List of common positive sentiment words list in movie reviews
- List of common negative sentiment words list in movie reviews

These lists of words did not come from the SST data so there is no risk of introducing bias in the model. After close analysis of SWN scores of these words, I found some words (37 negative and 16 positive) whose SWN scores I did not agree with. An over-ride mechanism was implemented in the processor for these words. Lists of these words can be found in the `SentiWordNetProcessor_NegHandling` class' __init__. Effectively, I am using a customized version of SentiWordNet for the SST task from hereon.

I was expecting the model to perform similarly on the positive reviews and improve on the negativ examples because of negation handling mainly. And a small improvement was also expected overall due to the domain-specific intervention.

**Results:** Major improvement in dev accuracy observed after this new swn processor was implemented. Dev accuracy climbed up to 0.540. 

</details> 

<h3>7. Should we trust the minBERT model or the SWN scores equally?</h3>

<details> 

**Explanation:** The model has two distinct components that it uses to predict sentiment of a sentence:
1. min BERT h_cls token’s 768 features
2. SWN’s 5 features (2 raw + 3 engineered)

The weight of these two sources is equal at the moment but in reality one source must be better or worse than the other. How much do I trust each source is the million dollar question. My idea is to use a learned gating system to let the model dynamically decide how much to trust SWN vs BERT features for sentiment classification.

**Implementation:** Instead of just concatenating SWN scores to h_cls, I introduced a simple learned gating system to let the model dynamically decide how much to trust SWN vs BERT features for each example.

It's a simple NN which takes in (h_cls + SWN) scores and outputs two weights (0.0 < weight < 1.0) using a sigmoid function. First weight for BERT features and second weight for the SWN scores.

Gating mechanism Architecture: (768+5) → 256 → 2

The bert_weight is multiplied element-wise to 768 BERT features, and the swn_weight is multiplied element-wise to the 5 SWN features before all are concatenated and mapped to the classification head.

Lastly, instead of a sigmoid function, a softmax was also tried to make the two weights sum up to 1.0 such that the two components, BERT and SWN, compete with each other for final prediction. But it reduced dev accuracy slightly so sigmoid was chosen ultimately. The sigmoid gives a 0-1 weight for each of BERT and SWN independent of each other.

My expectation from this change was a significant improvement in accuracy because now the model will weigh the two types of informations correctly instead of plain 50-50.

**Results:** Slight improvement in dev accuracy was observed but nothing dramatic.

</details> 

<h3>8. Can we add one more lexical database’s sentiment scores in addition to SWN scores to see if it improves the model?</h3>

<details> 

**Explanation:** The Valence Aware Dictionary and sEntiment Reasoner (VADER) scores are a document-level (sentence-level) polarity scores which describe positivity or negativity in sentiment of a text. There are 4 components returned by polarity_scores():

- pos: Positive sentiment proportion (0.0 to 1.0)
- neg: Negative sentiment proportion (0.0 to 1.0)
- neu: Neutral sentiment proportion (0.0 to 1.0): How much of the text is neutral/factual.
- compound: Overall sentiment score (-1.0 to +1.0)

Note: pos + neg + neu = 1.0 like the SWN scores

Paper: https://www.researchgate.net/publication/381650914_Understanding_Sentiment_Analysis_with_VADER_A_Comprehensive_Overview_and_Application

**Implementation:** Implementation was a much simpler activity because all the skeleton for the use of SWN scores was already set up and extracting VADER scores for each sentence was a trivial task. A new `VADERProcessor` class was written to extract these scores. The "neutral" vader score was not used due to same reason avg_obj_score from SWN was ommitted. Additionally, the gating mechanism was altered to cater to this novel, third type of features in addition to BERT and SWN. 

- OLD Gating mechanism Architecture: (768+5) → 256 → 2
- NEW Gating mechanism Architecture: (768+5+3) → 256 → 3

**Results:** As expected, the VADER scores enriched the model further with more information about sentiment from a new source so the dev accuracy rose to 0.544 which turned out to be the highest achieved in all the experiments.

</details> 

<h3>9. Adding more training data</h3>

<details> 

**Explanation:** Increasing training data would definitely help the model's performance so I tried to find open-source, fine-grained sentiment movie review datasets on the internet but was unsuccessful. The closest usable thing I found was the binary sentiment IMDB movie review dataset on kaggle https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews with 50000 examples. 

**Implementation:** I changed the sentiments from positive and negative to 4 and 0 respectively, changed the column names, appended the dataset to my training data. The simple data manipulation and saving can be seen in the `./data_extra/data_wrangling.ipynb`. I was not expecting good results because:
- The sentiments in the new data were not fine-grained which made the extra IMDB data different from my problem at hand which had 5 sentiments.
- Not all examples were usable for me because the average movie-review length was much bigger in the IMDB dataset which was causing out-of-memory errors during training so I was able to use only 1856 rows from the IMDB data for my training.

**Results:** At the end, after appending the extra data in training set the dev accuracy dropped to 0.503 so I reverted to my original training data. I did not add this section in the summary at the end because this was the last experiment and turned out unsuccessful.

</details> 

<h3>Summary of Experiments:</h3>

| Sno.| Experiment | Best Dev Accuracy |
|---|--------------|-------------------|
| 0 | Baseline | 0.521 |
| 1 | Remove attention masking from BERT | 0.520 |
| 2 | Hyperparameter tuning to solve overfitting | 0.519 (reverted) |
| 3 | SWN score (positive and negative) added to h_cls | 0.532 |
| 4 | Add dense layers between h_cls and classification head| 0.490 (reverted) |
| 5 | Engineer 3 new features from SWN positive and negative scores| 0.533 |
| 6 | Negation Handling and domain-specific knowledge added to SWN processor | 0.540 |
| 7 | Gating mechanism added to weight BERT model and SWN scores | 0.542 |
| 8 | VADER scores (positive, negative, compound) added to h_cls | 0.544 |

**Hyperparameters used for final model:**
- mode: `finetune`
- epochs: `10`
- learning rate: `1e-05`
- optimizer: `AdamW`
- dropout rate: `0.20`
- batch size: `64`

Note: The final model reaches peak dev accuracy of 0.544 in the 3rd epoch and saves itself. Beyond the 3rd epoch it just overfits to training data as highlighted in the experiment #2.





### Quora Question Pairs (QQP) - Question similarity

  

To continue with the Quora Question Pairs (QQP) project, my main goal was to surpass the initial baseline development accuracy of 0.870. For this part i made a few specific adjustments to the hyperparameters. The execution time for 6 epochs consistently remained at around 77 minutes (±1 min) across all runs, so this metric will not be a focus for further analysis.

  

I decided to increase the number of epochs to 6 and reduce the learning rate to 1e-5. My intention was to give the model more time to learn while taking smaller steps during the training process. This approach was successful, and my Hyperparamer-fine-tuned model achieved a new peak development accuracy of `0.883`. Although this wasn't the final, comprehensive set of changes, I decided to establish this as my new, personal baseline for the project. My next step will be to implement more significant modifications to try and improve on this new benchmark.

  

<h3>1. Multi-layer Perceptron Classifier for Paraphrase Detection</h3>

<details>

  

**Explanation:**

The original model used a single linear layer to classify paraphrases, which is a simple approach. The improved version replaces this with a **multi-layered classifier** to learn more complex patterns. This change is based on the principle that a deeper network can capture more nuanced relationships between the sentence embeddings, leading to better performance. The addition of a **GELU activation function** and an extra dropout layer brings non-linearity and helps prevent overfitting.


**Implementation:**

  
The single `nn.Linear` layer for the `paraphrase_classifier` was replaced with a `nn.Sequential` block. This new architecture consists of two linear layers, a GELU activation, and a dropout layer. The first linear layer transforms the concatenated BERT embeddings (`BERT_HIDDEN_SIZE`) into a hidden size of 768, while the second layer outputs the final logit. To ensure stable training, the weights of both linear layers were initialized using **Xavier uniform initialization**, and their biases were set to zero.

  

**Results**

  

The implementation of the new classifier resulted in a modest improvement in performance, with the accuracy increasing from 0.870 to 0.886.

</details>

<h3>2. Bi-Encoder Approach</h3>

<details>

  

**Explanation:**

  

The previous model's approach to paraphrase detection was limited. It concatenated the input sentences and processed them as a single sequence, relying on the BERT model to create a single, combined embedding. This method may not be optimal for capturing the individual nuances of each sentence and their relationship.

  

The new implementation improves on this by treating the sentences separately. By generating individual embeddings for each sentence (u and v), the model can explicitly compare them. The core of this improvement is the **S.I.A.M.E.S.E. (Sentence-pair similarity)** approach, which uses three distinct features for classification:

  

1. The embedding of the first sentence (u).

2. The embedding of the second sentence (v).

3. The absolute difference between the two embeddings (∣u−v∣).

  

This triplet of features allows the model to learn a better representation of the relationship between the sentences. It gives the model a more explicit signal about the magnitude of the differences between the two sentence vectors, which is a powerful indicator of their semantic similarity. This approach should lead to a **more robust and accurate model** for paraphrase detection.

  

**Implementation:**

  

The `predict_paraphrase` function was modified to first get the individual embeddings for each sentence, `u` and `v`, by calling `self.forward` on `input_ids_1` and `input_ids_2` separately. Dropout was applied to each embedding to prevent overfitting.

  

After obtaining the individual embeddings, the absolute difference `abs_diff = torch.abs(u - v)` was calculated. Finally, the three feature vectors—`u`, `v`, and `abs_diff`—were concatenated along the last dimension to form `combined_features`. This combined vector was then passed to the `paraphrase_classifier`, which was updated in the `__init__` function to accept an input size of `BERT_HIDDEN_SIZE * 3` to match the new feature representation. The output of the classifier is a single logit, which is then used for the final binary classification.

  

**Results:**

  

The implementation of the Siamese network with concatenated embeddings resulted in a decreased development accuracy, from a baseline of 0.883 to 0.803. This result is contrary to the expected improvement. A possible reason for this performance drop is the loss of contextual interaction between the two sentences that the original single-sequence approach provides. Although a Siamese network is generally effective, the direct concatenation and processing of both sentences by BERT in the baseline model might be capturing a crucial inter-sentence context that the separate-embedding approach misses. Additionally, the new, larger input to the classifier might be more difficult to train, and the model may have overfit on the training data, leading to a poorer generalization on the development set.

  

</details>

<h3>3. Comparison of Pooling Strategies: Mean, Max, Weighted, and Hierarchical Pooling</h3>

<details>

**Explanation:** 

To enhance my BERT model's performance in determining the semantic similarity of question pairs, I decided to replace the standard `[CLS]` token representation with several alternative pooling strategies. The `[CLS]` token is primarily pre-trained for next-sentence prediction, which I suspected wasn't the best way to capture a sentence's full meaning for my specific task. My goal was to generate more general, sentence-level vector representations. Each strategy I tested is based on a specific assumption about how a sentence's meaning can be derived from its individual token embeddings.

- **Mean Pooling:** My hypothesis for Mean Pooling was that a sentence's vector should be the average of all contextualized token embeddings. The assumption here is that every word in the sentence contributes equally to the overall meaning. By averaging these vectors, I expected to smooth out noise and irrelevant features, resulting in a robust and generalized representation. I believed this would better capture the semantic similarity between sentences since it considers information from the entire sentence, not just a single token.
    
- **Max Pooling:** In contrast, Max Pooling focuses on the most salient features of a sentence. The idea was to select the maximum value for each dimension across all token embeddings, retaining the most prominent features. I thought this would work well if the similarity between two sentences was determined by a few critical keywords.
    
- **Weighted Pooling (Attention Pooling):** I implemented this method with the assumption that not all tokens are equally important. I used a simple attention mechanism to assign a weight to each token's embedding, reflecting its relevance. I expected this strategy to provide a more precise representation by learning to focus on the most meaningful words while giving less importance to filler words.
    
- **Hierarchical Pooling:** This was my attempt at a hybrid method, combining the strengths of the `[CLS]` token and Mean Pooling. The `[CLS]` token captures global sentence-level information, while Mean Pooling provides an aggregated representation of all tokens. By concatenating these two vectors, I hoped to create a comprehensive representation that leverages both global and distributed features, ultimately leading to a superior performance.
    


**Implementation:** 

For each method, I modified the model's `forward` pass to return the corresponding pooled embedding.

- **Mean Pooling:** I implemented this by summing the token embeddings and dividing by the number of non-padding tokens, determined by the `attention_mask`. After pooling, I also applied layer normalization (`nn.functional.layer_norm`) to stabilize training.
    
- **Max Pooling:** For this, I masked padding token embeddings with a very small negative value (`-inf`) before applying `torch.max` along the sequence dimension to get the maximum value for each dimension.
    
- **Weighted Pooling:** I introduced an additional, trainable parameter (`attention_weights`). This vector is multiplied by the `last_hidden_state` token embeddings to calculate attention scores. The scores are then normalized using softmax and multiplied element-wise with the token embeddings before being summed.
    
- **Hierarchical Pooling:** I concatenated the `[CLS]` token vector (the first token in the sequence output) with the vector produced by mean pooling the remaining tokens. This created a longer vector for the classification layer to process.
    


**Results:** 

| Pooling Method | Accuracy (ACC) | 
| :--- | :--- | 
| Baseline (`[CLS]` token) | 0.883 | 
| Mean Pooling | 0.886 | 
| Max Pooling | 0.883 | 
| Weighted Pooling | 0.883 |
| Hierarchical Pooling | 0.882 |

With an accuracy of **0.886**, **Mean Pooling** was the only method that showed a measurable improvement over the baseline (`0.883`). The improvement is slight but confirms my hypothesis that capturing the full sentence information is a more effective strategy for this specific task compared to using the `[CLS]` token alone. The result highlights that a simple yet robust method can sometimes be the best practical solution.

Interestingly, both **Max Pooling** and **Weighted Pooling** showed no improvement. This suggests that focusing on a few prominent features or using a simple attention mechanism is not sufficient to capture the complex nuances of semantic similarity in the QQP data. It's possible that the semantic similarity in this dataset is embedded in the collective meaning of all words rather than in a few key terms.

The worst performance was observed with **Hierarchical Pooling**. The slight decrease in accuracy might indicate that combining the `[CLS]` token and Mean Pooling did not create the expected synergy. It's possible that the two vectors contain redundant information or that the concatenation results in an overly high-dimensional vector, which makes training the subsequent classification layer more difficult.

In conclusion, Mean Pooling was the best strategy. 

![Development over 6 Epochs](slurms/Bilder/qqp_pooling_plot.png)

The plot shows how the accuracy of each method changed over six training periods (epochs).  Given in red is the original baseline and the new self-made baseline. The graphs for `Cross Entropy Loss` and 'Bi-Encoder Approach' were not sown because the maximum accuracy is way below the baseline.
</details>

<h3>4. Adding a Contrastive Loss to Improve Paraphrase Detection</h3>

<details>

  

**Explanation**

  

While the standard binary cross-entropy loss trains the model to classify question pairs as paraphrases or not, it does not explicitly enforce this semantic similarity in the embedding space. My idea was to add a contrastive loss to the training process. This loss function would encourage the embeddings of similar questions to be closer to each other, while pushing the embeddings of dissimilar questions farther apart. By adding this as a regularizer, the model should not only classify correctly but also learn a more meaningful and structured embedding space, which I hoped would lead to better performance.

  

**Implementaion**

  

I introduced a secondary loss term, the **contrastive loss**, to the training loop. First, I modified the `predict_paraphrase` method to return the sentence embeddings (`u` and `v`) in addition to the classification logits. The embeddings were then normalized. A similarity matrix was computed using the dot product of these normalized embeddings. The contrastive loss was calculated using cross-entropy, where the model was trained to identify the matching pairs in the similarity matrix. This new loss term was then added to the original binary cross-entropy loss for classification, with a small weight of 0.1 to act as a regularizer. The total loss was then used for backpropagation.

  

**Results** 

  

The new implementation resulted in a decrease in accuracy from 0.883 to 0.826. This was an unexpected outcome, as the goal was to improve the model's performance by adding a contrastive objective.  It is also possible that the model struggled to learn both a good classification boundary and a well-structured embedding space simultaneously. A better approach might involve a different weighting of the contrastive loss or a more careful tuning of the temperature parameter, which was set to a fixed value of 0.05.

  

</details>

<h3>5. Different Model</h3>
<details>
I am testing a the newly pre-trained ´SimCSE´ model that was developed by Leonardo and me. For a more detailed breakdown, you can refer to his report.

This new model shows an improvement in performance compared to the baseline. The model shows one of the biggest improvment's with 0.887. It was also trained with a different amount of max batches, but these changes are negligible
</details>

  <h3>Combined Model</h3>
  <details>

  The main objective of this project was to improve an existing paraphrase detection model. Starting from an initial baseline accuracy of **0.870**, a hyperparameter tuning process (increasing epochs and lowering the learning rate) established an improved baseline of **0.883**. The goal was to surpass this new benchmark through new architectural and methodological.


### Promising Approaches

Two approaches had a positive impact on the model's performance. The first was the implementation of a **Multi-layer Perceptron Classifier**, which replaced the original model's simple linear layer with a deeper, non-linear structure. This change made it possible for the model to recognize more complex patterns, leading to an accuracy of **0.886**. This approach was successful because a more complex classification layer was better capture the relationships between sentence embeddings.

The second successful approach was **Mean Pooling**, which proved to be the most effective pooling strategy. Unlike the [CLS] token, which is primarily pre-trained for next-sentence prediction, Mean Pooling captured a representation of the entire sentence by averaging all token embeddings. This method also achieved an accuracy of **0.886**, suggesting that the overall meaning of a sentence is more critical for paraphrase detection than the features of a single token.


### Unexpected Setbacks

Other experiments led to a decrease in model performance. The **Bi-Encoder approach**, which processed the sentences separately, resulted in a significant drop in accuracy to **0.803**. The hypothesis is that this approach lost the crucial contextual interaction between the sentences that the original single-sequence BERT input provided. Although bi-encoder networks are often effective, in this specific case, direct inter-sentence context appears to be important for performance.

The addition of a **Contrastive Loss** was also disappointing. The goal was to bring the embeddings of similar sentences closer together, but instead, accuracy dropped to **0.826**. 


### The Challenge of Combination

Interestingly, the performance of **Hierarchical Pooling**, which combined the [CLS] token and Mean Pooling, was **0.882**, slightly worse than Mean Pooling alone. This suggests that combining approaches does not always lead to improvement. It's possible that the two vectors contained redundant information or that the higher dimensionality of the combined vector made subsequent classification more difficult.

A similar issue came up with the **Final Combined Model**, which showed no further improvement over the single best result (pre-training on an external dataset), which also achieved **0.887**. This raises the question if the different improvements were already addressing similar aspects of the problem. For example, the **pre-training** on external data might have already taught the model a robust semantic representation that subsequent architectural changes (like the Multi-layer Perceptron) only yielded marginal gains. The hypothesis is that a strong foundation, such as the one from pre-training, makes finer adjustments less impactful, as the biggest gains have already been achieved.

</details>

<h3>Summary of Experiments:</h3>

  

| Sno. | Experiment                           | Best Dev Accuracy |
| ---- | ------------------------------------ | ----------------- |
| 0    | Multi-layer Perceptron Classifier    | 0.886             |
| 1    | Bi-Encoder Approach                  | 0.803             |
| 2    | Mean Pooling and Layer Normalization | 0.886             |
| 3    | Contrastive Loss                     | 0.826             |
| 4    | Pre-training on an external dataset  | 0.887             |
| 5    | Final Combined Model                 | 0.887             |

  

Introduction

  

The following figure visualizes the performance of the different model variants during training. Each graph illustrates how the development accuracy (Dev Accuracy) evolved over the epochs, allowing for a direct visual comparison of each architecture's performance.

![Development over 6 Epochs](slurms/Bilder/qqp_changes.png)

  

Comparison of Model Performance During Training. This graph displays the development accuracy (Dev Accuracy), training accuracy (Train Accuracy), and training loss (Train Loss) for various model configurations over 6 epochs. Each line represents one of the tested architectures. The use of different markers (e.g., circles for development accuracy, triangles for training accuracy) helps distinguish between the metrics. The red dashed line indicates the initial baseline accuracy of 0.870.

### Semantic Textual Similarity (STS) - Measuring text meaning similarity

### Paraphrase Type Detection (PTD) - Identifying paraphrase types and relationships

### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types




--------------------------------------



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