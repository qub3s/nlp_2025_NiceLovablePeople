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
- Pretrain SimCSE on NLI datasets (Support)

Leonardo Christian da Camara Silva:
- Semantic Textual Similarity (STS)
- Pretrain SimCSE on NLI datasets (Lead)
- bonus: Paraphrase type detection with bert (PTD-bert)

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

  

I decided to increase the number of epochs to 6 and reduce the learning rate to 1e-5. The Hyperparamer-fine-tuned model achieved a new peak dev accuracy of `0.883`. Although this wasn't the final, comprehensive set of changes, I decided to establish this as my new, personal baseline for the project. My next step will be to implement more significant modifications to try and improve on this new benchmark.

  

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

  

This triplet of features allows the model to learn a better representation of the relationship between the sentences. It gives the model a more explicit signal about the magnitude of the differences between the two sentence vectors, which is a powerful indicator of their semantic similarity. This approach should lead to a better and more accurate model for paraphrase detection.

  

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

![Development over 6 Epochs](qqp_extra/Bilder/qqp_pooling_plot.png)

The plot shows how the accuracy of each method changed over six epochs.  Given in red is the original baseline and the new self-made baseline. The graphs for `Cross Entropy Loss` and 'Bi-Encoder Approach' are not sown because the maximum accuracy is way below the baseline.

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

This is the model used in the final commit.

**Expectation**

Based on the results of the experiments, The goal was to create a optimized model by combining the most successful components. The goal was to leverage the strengths of each improvement to achieve a performance that surpassed the highest individual accuracy. I chose to use three key modifications: **Mean Pooling** for a more comprehensive sentence representation, a **Multi-Layer Perceptron (MLP) classifier** for a more powerful and non-linear classification head, and the  **new model** that had already demonstrated improved performance. My hypothesis was that the combination of a richer input representation and a stronger classification layer, built on an already-strong foundation, would result in a good leap in accuracy. 


**Results**

The final combined model achieved a peak development accuracy of **0.887**. While this result represents the highest accuracy achieved during the project, the improvement over the previous best of 0.887 from the individual Mean Pooling and MLP experiments was minimal. This outcome suggests a case of diminishing returns, where adding incremental improvements to an already optimized model bring no longer significant gains. The combination of features did not produce the expected effect. This implies that the current architecture and dataset may have a performance ceiling, and further  improvements would require a more fundamental change, such as migrating to a more advanced pre-trained model or different training paradigms.

</details>

<h3>Summary of Experiments:</h3>

<details>

  The main objective of this project was to improve an existing paraphrase detection model. Starting from an initial baseline accuracy of **0.870**, a hyperparameter tuning process (increasing epochs and lowering the learning rate) established an improved baseline of **0.883**. The goal was to surpass this new benchmark through new architectural and methodological.


### Approaches

Two approaches had a positive impact on the model's performance. The first was the implementation of a **Multi-layer Perceptron Classifier**, which replaced the original model's simple linear layer with a deeper, non-linear structure. This change made it possible for the model to recognize more complex patterns, leading to an accuracy of **0.886**. This approach was successful because a more complex classification layer was better capture the relationships between sentence embeddings.

The second successful approach was **Mean Pooling**, which proved to be the most effective pooling strategy. Unlike the [CLS] token, which is primarily pre-trained for next-sentence prediction, Mean Pooling captured a representation of the entire sentence by averaging all token embeddings. This method also achieved an accuracy of **0.886**, suggesting that the overall meaning of a sentence is more critical for paraphrase detection than the features of a single token.


### Setbacks

Other experiments led to a decrease in model performance. The **Bi-Encoder approach**, which processed the sentences separately, resulted in a significant drop in accuracy to **0.803**. The hypothesis is that this approach lost the crucial contextual interaction between the sentences that the original single-sequence BERT input provided. Although bi-encoder networks are often effective, in this specific case, direct inter-sentence context appears to be important for performance.

The addition of a **Contrastive Loss** was also disappointing. The goal was to bring the embeddings of similar sentences closer together, but instead, accuracy dropped to **0.826**. 


### Combination

Interestingly, the performance of **Hierarchical Pooling**, which combined the [CLS] token and Mean Pooling, was **0.882**, slightly worse than Mean Pooling alone. This suggests that combining approaches does not always lead to improvement. It's possible that the two vectors contained redundant information or that the higher dimensionality of the combined vector made subsequent classification more difficult.

A similar issue came up with the **Final Combined Model**, which showed no further improvement over the single best result (pre-training on an external dataset), which also achieved **0.887**. This raises the question if the different improvements were already addressing similar aspects of the problem. For example, the pre-training on external data might have already taught the model a robust semantic representation that subsequent architectural changes (like the Multi-layer Perceptron) only yielded marginal gains. The hypothesis is that a strong foundation, such as the one from pre-training, makes finer adjustments less impactful, as the biggest gains have already been achieved.

  

| Sno. | Experiment                           | Best Dev Accuracy |
| ---- | ------------------------------------ | ----------------- |
| 0    | Multi-layer Perceptron Classifier    | 0.886             |
| 1    | Bi-Encoder Approach                  | 0.803             |
| 2    | Mean Pooling and Layer Normalization | 0.886             |
| 3    | Contrastive Loss                     | 0.826             |
| 4    | Pre-training on an external dataset  | 0.887             |
| 5    | Final Combined Model                 | 0.887             |


The following figure visualizes the performance of the different model variants during training. 

![Development over 6 Epochs](qqp_extra/Bilder/qqp_changes.png)


This graph displays the Dev accuracy for various model configurations over 6 epochs. Each line represents one of the tested architectures. The red dashed lines indicates the initial baseline accuracy of 0.870 and the self-made baseline. Only the additions that made it over the baseline are shown.

</details>

### Semantic Textual Similarity (STS) - Measuring text meaning similarity

This chapter investigates methods for enhancing semantic textual similarity (STS) by leveraging the SimCSE framework for contrastive pre-training. In Figure 1 is a rough sketch displayed, of the implemented approach which begin with a BERT Base uncased model. Subsequently it is pre-trained using supervised SimCSE on NLI datasets. This pre-trained model serves as a foundation for subsequent fine-tuning on STS data using several methodologies: a standard SBERT architecture with MSE loss, a SimCSE model with contrastive loss, and a novel combined approach (SBS+SimCSE) that uses a weighted sum of both objective functions. The following sections detail the implementation and results of these experiments.

<details>
Remarks: For all plots 15 different seeds were used to compute 95% confidence intervalls. The baseline of the model with minBert had a 0.371 dev correlation in Part-01.


![STS task Framework](STS_Plots_Pretrain/graphic.png)
*Figure 1: Framework for STS improvements*

</details>

<h3>1. Pretrain the given Basemodel with SimCSE</h3>
<details>


**Explanation:** 

The Semantic Textual Similarity (STS) task requires predicting a fine-grained similarity score between 0 and 5 for sentence pairs. While pre-trained BERT Base model offer a strong foundation of linguistic knowledge, it is not optimized for semantic similarity tasks. To bridge this gap, this work impliments the SimCSE framework, as introduced by Gao et al. [1], which employs contrastive learning to significantly enhance sentence embeddings. Both the unsupervised and supervised methods described in the paper are implemented here.

In the unsupervised approach SimCSE learns sentence embeddings through contrastive learning across two distinct settings. Here one generates positive pairs by passing identical sentences through the same encoder with different dropout masks applied stochastically during training. This teaches the model to produce invariant representations for semantically equivalent inputs, pulling them closer in the embedding space despite minor variations induced by noise. The SNLI [2] and MNLI [3] dataset were used to train the unsupervised approach.

The supervised approach also leverages the rich semantic signals in Natural Language Inference (NLI) datasets like SNLI and MNLI. It uses labeled entailment pairs (premise-hypothesis) as strong positives. Crucially, it also employs contradiction pairs as hard negatives, providing a direct signal that helps the model not only learn similarity but also sharpen its ability to discriminate between related and unrelated sentences.


**Implementation:**

This implementation enhances the original SimCSE technique [1] by instituting several methodological modifications. The pooling method was expanded to include mean and max pooling options in addition to CLS token pooling. Mean pooling, on the other hand, improved semantic representation by averaging token embeddings. A non-linear projection head transforms the standard 768-dimensional embeddings into a 256-dimensional space, aligning with SimCSE best practices for enhanced representation learning.

The contrastive loss function got a lot better thanks to margin separation methods and thorough normalisation. This made it easier to tell the embedding clusters of different texts apart. For supervised training, the system uses Natural Language Inference data and hard negatives from contradiction pairs. This gives stronger signals for learning than methods that are only unsupervised.

Automatic mixed precision training made calculations faster by using less memory and keeping numbers stable through gradient scaling. Gradient accumulation methods let one use batch sizes that are bigger than what most GPUs can handle. This makes training more stable. To speed up convergence, the learning rate schedule uses linear scheduling and warmup periods. And at the end, an early stopping system with patience monitoring stops training when the performance on the development set stops getting better.


**Results:**

Model performance is evaluated on the Semantic Textual Similarity (STS) benchmark using Spearman's rank correlation coefficient. This metric measures how well the cosine similarity between sentence embeddings aligns with human-annotated similarity scores.

The supervised SimCSE model achieved a Spearman correlation of 0.8216, significantly outperforming the unsupervised variant, which reached 0.6824. These results demonstrate that it is an improvement to incorporate labeled natural language inference data into the training process.

</details>

<h3>2. SBert Finetuning</h3>
<details>


**Explanation:**

SBERT enhances the STS task by generating high-quality sentence embeddings that can be efficiently compared using cosine similarity. In comparement to standard BERT, which requires pairwise computations and is computationally expensive, SBERT uses a siamese network structure fine-tuned the given STS dataset. This allows it to produce semantically meaningful embeddings that significantly outperform traditional methods like averaging BERT outputs or using GloVe embeddings [4]. As a result, SBERT reduces inference time from hours to seconds while maintaining or improving accuracy on STS benchmarks.


**Implementation:**

This implementation realizes the SBERT architecture through a modular design centered on a shared BERT encoder with tied weights. The core of the system lies in its mean pooling strategy, which computes sentence embeddings by averaging the output token vectors while dynamically accounting for variable input lengths through mask-based normalization. The resulting embeddings are L2-normalized before similarity computation to ensure stable cosine similarity measurements within a unit sphere.

The training regime uses a mean squared error objective function, directly optimizing the model to regress towards continuous similarity labels. The framework incorporates a training loop with gradient accumulation and optimizer scheduling, which allows for effective batch processing. A linear scaling operation transforms the normalized cosine similarity scores to the target evaluation range, aligning the model's output to the output [0,5].


**Results:**

To evaluate the effect of pretraining, the SBERT model was trained with and without a pretrained SimCSE encoder. The model got a development set pearson correlation of 0.582 when it finetuned based in the Bert base model and trained on the whole STS training dataset. In striking contrast, using the encoder with pretrained SimCSE weights and fine-tuning it on only with 320 STS phrase pairings gave a far better correlation of 0.843.

To further examine this, the performance of the SimCSE-enhanced SBERT was tested across various training dataset sizes. Figure 2 shows the findings, which demonstrate a clear pattern: Good performance starts with very little data, with a correlation of 0.8 corr with just 160 corr sentences. The best performance occurs between 320 and 2880 sentences, when the correlation is around 0.82 corr. But when trained on bigger parts of the dataset, performance reduces quickly to around 0.72 corr.

A possible explanation is that the pretrained SimCSE model is already good at generating sentence embeddings that have semantic significance because of its contrastive learning goal. So, to align these embeddings for the semantic textual similarity task, you just need a little quantity of STS data that is particular to the job. Fine-tuning on a bigger and perhaps noisier dataset might make the model overfit or forget the general-purpose representations that made it work well in the first place. This would make it perform worse on the evaluation benchmark.

![](STS_Plots_Pretrain/batch_vs_correlation_ci_log.png)
*Figure 2: SBErt Performance using the pretrained SImCSE model as Base Bert model (Batchsize=32 with 5719 training sentence pairs)*

</details>

<h3>3. SimCSE Finetuning</h3>
<details>


**Explanation:**

This implementation adapts the SimCSE framework for supervised finetuning. It uses annotated sentence pairs to direct the contrastive learning process, in contrast to the unsupervised approach [1]. In order to create positive pairs, the model learns by comparing each sentence to itself after it has been run twice through the encoder using various dropout masks. It employs other sentence pairs in the batch as negatives at the same time. By combining human-rated similarity labels with contrastive representation learning, it improves the capacity and can generate sentence embeddings for the similarity task.


**Implementation:**

The system builds on the BERT-based encoder that has been fine-tuned using a custom contrastive objective or the Bert-based uncase. The SimCSE model produces two embeddings for each sentence in a pair within a batch by employing stochastic dropout, resulting in natural variations that act as positive examples. While a contrastive loss function promotes similarity between these augmented views of the same sentence, it diminishes similarity to all other sentences in the batch. During the training process, STS data is iterated over and weights are updated via backpropagation, with no explicit similarity scores used during contrastive updates.


**Results:**

Similar to the SBERT approach, the SimCSE framework was used with the BERT-base model and the SimCSE pretrained BERT model. This resulted in a 0.742 Pearson correlation with the BERT-base model and a 0.815 correlation for the pretrained SimCSE model. The BERT-base model was fine-tuned on the full STS dataset, whereas the pretrained SimCSE model was fine-tuned with 960 STS sentence pairs.

The already pretrained SimCSE model, like in the SBERT fine-tuning, needs only a fraction of the data to achieve high performance. This is further investigated in Figure 3, where the size of the training data is measured against the framework's performance. It starts with a correlation of 0.7775 for 320 sentence pairs, peaks with a correlation of 0.787 at 960 sentence pairs, and then stays approximately at a correlation of 0.75. This could be explained by the fact that after the model has learnt as much as it can from the small amount of supervised data, more data offers little benefit and may even introduce noise that marginally impairs performance.

If one compares the SBERT fine-tuning approach with the SimCSE fine-tuning approach, both show significantly better correlations with the applied pretrained SimCSE model. A difference arises when comparing correlations obtained without using the pretrained SimCSE model. Here, SBERT at 0.582 is significantly lower than SimCSE fine-tuning at 0.724 correlation, while for the framework with the pretrained SimCSE model, the SBERT structure has a better correlation. This can be explained by the fact that the SBERT architecture is better at utilising high-quality pre-existing embeddings, whereas the SimCSE fine-tuning method performs better at baseline when starting from a generic BERT model because of its built-in contrastive objective.

![](STS_Plots_Pretrain/simcs_batch_vs_correlation_ci.png)
*Figure 3: SBert performance for different training data size (Batchsize=32 with 5719 training sentence pairs)*
</details>

<h3>4. SBert + SimCSE Finetuning</h3>
<details>


**Explanation:**

The approach combines the strengths of two powerful sentence embedding methods: SBert and SimCSE. As mentiond above: SBert is fine-tuned supervised, and so it may overfit to label noise and lose generalization. SimCSE, on the other hand, uses contrastive learning to pull semantically similar sentences closer in the embedding space while pushing dissimilar ones apart, improving robustness and representation quality. By optimizing both objectives together during fine-tuning, this could lead to a framework which offers SBert’s task-specific performance while preserving the generalization benefits of SimCSE.


**Implementation:**

The implementation integrates SBert and SimCSE into a unified training pipeline. It uses a shared BERT-based encoder to produce two types of embeddings: one for SBert (mean-pooled token embeddings) and one for SimCSE (CLS token output). During training, it computes two losses: a mean squared error (MSE) loss for SBert’s regression task and a contrastive loss for SimCSE. The contrastive loss is applied to two different dropout-masked versions of the same sentence, encouraging invariance and better alignment. The total loss is a weighted sum of both objectives, controlled by a hyperparameter alpha. The implimentation uses a combined dataloader that serves STS triplets (sentence1, sentence2, label) and optimize the model end-to-end. This allows the model to simultaneously learn from explicit similarity labels and implicit contrastive signals.


**Results:**

![](STS_Plots_Pretrain/nlp_sber_simcse.png)
*Figure 4:Different weighting between SBert and SimCSE loss and the effect on fine-tuning performance on the STS dataset*

The best development set correlation was achieved with an alpha value of 0.975 (see figure 4), indicating that the model benefits most from the SimCSE contrastive loss, with only a small contribution from the SBERT regression loss. This configuration yielded a correlation that was 0.4% higher than the best result from standard SBERT fine-tuning. So its a little improvement to the state of the art approach.

For most alpha values below 0.975 (alpha < 0.975), the correlation remained stable within a high range of 0.824 to 0.826, showing a slight linear increase. However, beyond this point, the correlation weakened significantly, dropping to 0.814 for alpha = 1.0 (using only the SimCSE loss). This can be explained by the fact that a small amount of supervised signal from the STS labels is crucial for guiding the contrastive learning process on the specific task; without it, the model lacks the necessary directional cues for optimal performance on semantic similarity.

Notably, even at alpha = 1.0, the implementation of the SimCSE-only model outperformed the standalone SimCSE model from the previous experiments. This indicates that the initial pre-trained base model and the architectural setup of the combined framework provide a stronger foundation for contrastive learning than a standard implementation.


![](STS_Plots_Pretrain/simcse_sbert_batches.png)
*Figure 5: SBert+SimCSE performance for different training data size(Batchsize=32 with 5719 training sentence pairs)*

In addition, experiments with varying training data sizes revealed a behavior similar to the SBERT case: performance peaked at  640 sentence pairs. Using more data for fine-tuning resulted in decreased correlation, suggesting that excessive task-specific fine-tuning can weaken the general, high-quality embeddings obtained from pre-training.

</details>

<h3>5. Summary of Experiments and literature:</h3>
<details>

**Hyperparameters used for final SimCSE + SBert Model with pretrained SImCSE:**
- mode: `finetune`
- epochs: `7`
- learning rate: `2e-05`
- optimizer: `AdamW`
- dropout rate: `0.30`
- batch size: `64`
- pretrained_simcse: `best_model_epoch3_corr0.8216.pt`
- number of batches: `10`
- alpha: `0.975`

Note: The final model reaches a peak dev accuracy of 0.847 when executed from the test function. The best model during training on the dev STS data achieved a score of 0.826. This explains the approximately 2% difference in dev correlation observed in the figures compared to the best values, as the plot uses the dev correlations evaluated during training. This issue also appeared in the Part 01 submission.

**Literature**

#### 1. SimCSE
- **Citation**:
  Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings.

#### 2. SNLI
- **Citation**: 
  Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A Large Annotated Corpus for Learning Natural Language Inference.

#### 3. MultiNLI
- **Citation**:
  Williams, A., Nangia, N., & Bowman, S. R. (2018). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference.

#### 4. Sentence-BERT
- **Citation**:
  Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
  
 </details>

 | Sno.| Experiment | Best Dev Correlation |
|---|--------------|-------------------|
| 0 | Baseline | 0.371 |
| 1 | Baseline  with Pretrained SimCSE| 0.809 |
| 2 | SimCSE without Pretrained SimCSE| 0.724 |
| 3 | SimCSE with Pretrained SimCSE | 0.815 |
| 4 | SBert without Pretrained SimCSE | 0.582 |
| 5 | SBert with Pretrained SimCSE | 0.843 |
| 6 | SimCSE + SBert without Pretrained SimCSE | 0.750 |
| 7 | SimCSE + SBert with Pretrained SimCSE | 0.847 |

### Paraphrase Type Detection

<h3> 1. What causes the high discrepancy between accuracy and Matthews Correlation Coefficient? </h3>
<details> 

**Explanation:** <br>
While accuracy and the Matthews Correlation Coefficient assess different aspects of model performance, it was still surprising to us that a model can have over 90% accuracy and a ~0 Matthews Correlation. To investigate this phenomenon, we implemented additional metrics (precision, recall, f1) that are more intuitive to us. Additionally, we looked at the occurrence rates of the label and at the predictions we submitted in the first stage.

**Implementation:** <br>
To implement the metrics, we calculated the TP, FP, TN, and FN values for all samples and then used the respective formulas.

**Results:** <br>
The high Precision shows that the majority of the positively predicted examples are correct; however, the low Recall suggests that the model fails to detect many samples as positive. The manual inspection of the results and the plotting of the class distribution revealed a likely hypothesis for why this is. The results showed that only a few classes were predicted (almost) all the time. The class distribution showed that these were the classes with the most frequent occurrence in the dataset. The dataset in general was very unevenly distributed between the labels, half a dozen had less than 100 samples, the smallest label only having four occurrences, while others had 2,705 occurrences (in a dataset of 2,730 samples). This poses a problem at both ends of the spectrum: for the very small classes, there is not enough information for the model to get a good feel for "what they are," and for the large classes, there are not enough negative samples. 

![alt text](imgs/fig_1.png)
</details>

<h3> 2. Establishing a Reasonable Baseline for Improvement</h3>
<details> 

**Explanation:** <br>
In the first stage, we were given a set of parameters with which we were supposed to train the model. Before running any experiments, we wanted to set a better baseline to judge whether subsequent methods really show an improvement. For this, we tested different batch sizes and learning rates and implemented Early Stopping to avoid having to worry about the correct number of epochs.

**Implementation:** <br>
We implemented a minimal early stopping class which is initialized with a patience value (how many epochs without improvement are allowed) and a path where the model is stored. Furthermore, there is a "call" function, which takes the model and the current validation score as input. If there is an improvement in the validation score, the model snapshot is saved; otherwise, the patience counter is increased. The class has an early stop variable which can be used to check whether the training should be stopped. After these changes, we expected a moderate increase in performance.

**Results:** <br>
The parameters remained similar to the ones we used for the previous stage (lr: 1e-5, batch size: 64), but the usage of early stopping made a large difference. We decided to use a patience value of 5 for all subsequent experiments. The Matthews Correlation increased from ~0 in the first stage to 0.09. This performance increase was larger than expected and indicates that the training time for stage one was too short.

*Additionally we want to note here that during the testing the training results were vastly different depending on small parameter or seed changes. This might effect the interpretability of results later on. For example we had 0.03 changes in MCC when changing batchsize from 64 to 32.*

</details>

<h3> 3. Is the model not complex enough to solve the problem?</h3>
<details> 
  
**Explanation:** <br>
The model from the first stage had a single linear layer. First, we wanted to check if additional linear layers might increase performance. The idea was that the model might only classify examples along the "most valuable" boundaries since it only had the ability to use a linear boundary, and thus a model that can differentiate along more complex boundaries might perform better.

**Implementation:** <br>
In the implementation, we replaced the single layer with 2 (768 -> 64 -> 26) or 3 (768 -> 128 -> 64 -> 26) layers respectively. Between the layers, ReLU and Dropout (0.2) were used except in the last layer where the sigmoid function remained as before. We expected minimal increase in the results at best.

**Results:** <br>
The results were much worse than the baseline, with the three-layer version (0.022) being better than the two-layer one (0.009). Interestingly, the 2-layer version showed a higher precision and lower recall than the baseline and the 3-layer model. This indicates that the model probably instantly overfits if you add additional linear layers or potentially is already overfitting. 
</details>

<h3> 4. Do the ideas behind Focal Loss help increase the performance in this task?</h3>
<details> 

**Explanation:** <br>
To try to address the shortcomings of the current model, we used an idea from computer vision called Focal Loss. It is a loss function which uses two separate ideas to address class imbalance in dense object detection tasks. The alpha parameter tries to mitigate the class imbalance by weighting the loss in favor of rarely occurring classes. The strength of this weighting is determined by the alpha parameter(s). The gamma parameter tries to focus on "hard" samples instead of being sidetracked by the easy ones. Both should work in our favour, as we identified the class imbalance as the major problem in this task, so if the weighting works well, it might improve the performance significantly. The gamma parameter is more of a wildcard, but the large labels are predicted with high confidence, so averting the loss away from those would also be good. To test which of these ideas works, we tested them individually. For the alpha values we tested $\frac{1}{x}$, $\frac{1}{\sqrt(x)}$, $\frac{1}{^3\sqrt(x)}$, $\frac{1}{^4\sqrt(x)}$, $\frac{1}{^5\sqrt(x)}$ and for gamma values we tested 1 to 5.

**Implementation:** <br>
We implemented a Focal_Loss class as a PyTorch loss (nn.Module), which just implements the mathematical formulation of the [Focal Loss](https://arxiv.org/pdf/1708.02002).

**Results:** <br>
Both ideas, the balancing alpha parameter and the gamma parameter, seem to work fairly well. The best value for the gamma parameters was achieved at a gamma of 3, which resulted in an MCC of 0.12, surpassing the 0.09 value of the baseline. The alpha parameter showed a similar behaviour, peaking at "alpha_1/sq4" and an MCC of 0.126.

</details>

<h3> 5. Does oversampling help mitigate the effects of class imbalance?</h3>
<details> 

**Explanation:** <br>
One of the methods to tackle class imbalances is oversampling. Oversampling is the process of duplicating samples to balance out the class frequencies. We tried a method where we assign a score to every sample based on its labels and then sample from this list based on the scores. The scores were calculated on the inverse frequency ($\frac{1}{X}$) of the labels (or their square roots $1/\sqrt{X}$).

$value = \sum^0_{26} target_x * freq_x / \sum^0_{26} target_x$

Through this, in practice, we do not create any copies of the data, but during sampling, we will pull more data which consists of rare classes. 

**Implementation:**  <br>
We implemented this by writing the class Weight_based_sampler, which inherits from the PyTorch class torch.utils.data.Sampler. It calculates the values as described above, turns them into probabilities, and then samples from that distribution using np.random.choice. 

**Results:** <br>
We had high hopes for oversampling, but this method did not work well. None of the parameters that were tested reached the MCC value of the baseline, even though they came very close. This indicates that this type of sampling is not suitable for this task. 

</details>

<h3> 6. Putting together what works!</h3>
<details> 

**Explanation:** <br>
At the end, we tried to combine the methods that did show potential into a single model. For that, we used both parts of the focal loss (alpha and gamma). We went with the best values for both (alpha: 1/sq4 and gamma: 3). We tested a variety of learning rates, batch sizes, and seeds. Furthermore, we experimented with changing the decision boundary between the classes, which was by default at 0.5, to different numbers.

**Results:** <br>
On our validation set, we received a wide range of results depending on the input parameters. The best result we managed to achieve for the default seed was an MCC value of 0.467 (20% validation set size). However, we decided to submit a different model with an MCC of 0.243, because the "best" model configuration performed very poorly on other seeds. Changing the decision boundary made a huge difference in some configurations; however, in others, it made barely any difference at all. Furthermore, there was no clear observable trend as to whether a lower value is preferable across different experiments.

</details>


*All experiments were run on 1e-5 learning rate, patience 5, and batch size 64.*

| Experiment | Name | precision | recall | f1 | Correlation | 
|--|--|--|--|--|--|
|2|baseline|0.802|0.579|0.673|0.09|
|3|2_layers|0.963|0.307|0.466|0.009|
|3|3_layers|0.869|0.409|0.627|0.022|
|4|alpha_1|0.790|0.565|0.659|0.014|
|4|alpha_1/sq2|0.779|0.595|0.674|0.077|
|4|alpha_1/sq3|0.792|0.583|0.672|0.104|
|4|alpha_1/sq4|0.817|0.575|0.675|0.126|
|4|alpha_1/sq5|0.857|0.531|0.656|0.099|
|4|gamma_1|0.814|0.534|0.631|0.097|
|4|gamma_2|0.837|0.558|0.670|0.118|
|4|gamma_3|0.827|0.547|0.658|0.12|
|4|gamma_4|0.857|0.531|0.656|0.086|
|4|gamma_5|0.841|0.551|0.666|0.101|
|5|1/x|0.785|0.576|0.664|0.026|
|5|1/sq2(x)|0.843|0.524|0.647|0.078|
|5|1/sq3(x)|0.820|0.565|0.669|0.064|
|5|1/sq4(x)|0.838|0.520|0.642|0.089|
|5|1/sq5(x)|0.837|0.528|0.647|0.075|
|6|best_model|0.889|0.792|0.838|0.467|
|6|submitted_model|0.783|0.666|0.720|0.243|

**Besides the class imbalance, the biggest problem in this task turned out to be the unreliable results. If we had invested more time into this project, this would be another core priority that would need to be tackled.**

### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types

**Explanation** <br>
Paraphrase Type Generation is a specialisation of the Paraphrase Generation task.    
A paraphrase is a rewording of a sentence without changing the semantic meaning of it.
Wahle et al. (Wahle et al. 2023) introduced a new approach to incorporate paraphrase types into models for paraphrase detection and generation in their paper Paraphrase Types for Generation and Detection. Models that are trained without considering paraphrase types fail to understand how they alter sentences. Therefore, training them with these types can lead to improved language understanding. <br>
Their dataset (https://huggingface.co/datasets/jpwahle/etpc) is used for this task, which incorporates paraphrase types based on the paper ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negati by Kovatchev et al. (Kovatchev et al., 2018). <br>

**Motivation** <br>
Training a model to generate paraphrases can be used in multiple ways. It could be used to generate synthetic data, which could be used to train a detection model. Li et al. (Li et al., 2018) mentions "retrival based question answering, semantic parsing, query reformulation in web search and data augmentation for dialog systems" as possible applications.

**Implementation** <br>
In this task the input is a combination of the input sentence made of a sequence of words $W = [w_1, w_2,...]$, the locations of segments $L = [l_1, l_2,...]$ where the paraphrase should occur and the paraphrase type ids $T = [t_1, t_2, ...]$. This information gets concatenated with a special token to indicate separation and then tokanized by the bart_large tokenizer by facebook to be ready for the Bart model (https://huggingface.co/facebook/bart-large). 
The output should be another sentence made off a sequence of words $O = [o_1, o_2, ...]$ with the same semantic meaning as $W$, but not the same sequence of words. <br>

**File Structure** <br>
- bart_generation_baseline.py: baseline model training implementation of part 1
- bart_generation_earlystopping.py: implementation of early stopping as described below
- bart_generation_improvement.py: includes model training implementation of part 2 with all improvements described below
- bart_generation_RL.py: class for reinforcement learning setup as described below

<h3> 1. Setup for Improvements: Hyperparameters and Early Stopping </h3>
In preparation for work on improvements, the baseline training has been optimised. This involved testing different hyperparameters and combinations of these, as well as introducing early stopping based on the new metric, the penalised BLEU score.

<details> 

<h4> Early Stopping </h4>

**Explanation:** <br>
In my experiments, I applied early stopping based on the penalized BLEU score of the validation set. This approach allowed me to train for longer without overfitting on this evaluation metric. 

**Implementation:** <br>
The implementation is provided in bart_generation_early_stopping.py and is adapted from the version originally used for the paraphrase type detection task. It monitors the validation score during training and keeps track of the best-performing model. After each epoch, the current score is compared to the best score so far. If the score improves, both the best score and model are updated. If no improvement occurs within a specified number of epochs, called patience value, training stops early, and the best model is saved while the relevant epoch and score information is being logged.

**Results:** <br>
Although computing BLEU at every epoch is computationally expensive, it provided a more reliable signal of progress than validation loss. In fact, the loss fluctuated more strongly and suggested overfitting prematurely, whereas the BLEU score steadily improved. For this reason, the extra computation required to track BLEU appears worthwhile. <br>
During the later tests different training set ups required different amounts of epochs, which would have been a struggle to manually track and adjust to.

<h4> Weight Decay </h4>

**Explanation:** <br>
The weight decay of the AdamW optimiser was used as described in the paper Decoupled Weight Decay Regularization by Loshchilov and Hutter (Loshchilov and Hutter, 2019).

**Implementation:** <br>
The weight decay is already implemented in the AdamW optimiser in the optimiser.py file and could simply be added as a parameter.

```python
optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
```

**Results:** <br>

Adding a weight decay of 0.01 to the AdamW optimiser, as is often the default weight decay, improved the penalised bleu score compared to the base line by around 0.6091 after 5 epochs with a value of 2.5911. When running with early stopping (patience=5) it stopped after epoch 37 with the following values:
```
Bleu Score after epoch 37: 21.9025
Negative Bleu epoch 37: 24.3992
Base Score epoch 37: 46.6790
```

<h4> Batch Size </h4>

**Explanation:** <br>
Different batch sizes were explored to find their influence on the training.

**Implementation:** <br>
Three smaller batch sizes were tested in comparison to the baseline batch size, as they showed signs of improving the penalty BLEU score. A larger batch size could be used to stabilise training if the learning rate or penalty loss causes instability, but this was not specifically tested in this section as it did not improve the penalised BLEU score. All other training parameters were kept constant across runs. To simplify testing, reduce computational cost, and enable a larger number of runs, the number of epochs was limited to 25. This was sufficient to observe general trends, while more promising configurations were later trained for the full duration.
- Batch Size options: {32, 16, 8, 4}
- Learning Rate: 1e-05
- Patience: 5
- Weight Decay: 0.01
- Loss: cross entropy loss
- Epochs: 25

**Results:** <br>
With a batch size of 32, training was stable and continued to improve until it was stopped at epoch 25. At this point, the penalised BLEU score reached 17.9334, with only slight overfitting indicated from epoch 11 onwards. <br>
Reducing the batch size to 16 accelerated learning, reaching a penalised BLEU score of 21.0785 after 25 epochs. However, overfitting began earlier, around epoch 7, and the validation loss fluctuated more. <br>
A batch size of 8 achieved the highest peak performance, with a maximum penalised BLEU score of 24.0398 at epoch 18. Yet, this setting was considerably less stable, showing strong fluctuations and overfitting after epoch 4, suggesting that higher patience would be required to make use of this configuration, as it stopped already at epoch 18. <br>
With a batch size of 4, instability was even more pronounced, with the penalised BLEU score rising quickly in the early epochs (e.g., 11.4046 at epoch 5) but severe overfitting in the validation loss. It achieved a penalised BLEU score of 25.2094 in epoch 17, but this appears to be an anomaly given the significant fluctuations; it decreased immediately afterwards to 20.3324 and 22.9765 in the subsequent epochs. <br>
Overall, smaller batch sizes accelerated improvements in the penalized BLEU score but introduced instability and earlier overfitting, while larger batch sizes (32) provided more consistent and stable learning. For further testing, batch size 32 was preferred for reliability, with batch size 16 reserved for potential fine-tuning in final experiments.

<h4> Learning Rate </h4>

**Explanation:** <br>
Different learning rates were explored to find their influence on the training.

**Implementation:** <br>
In comparison to the default learning rate of 1e-5 used in the base model, three alternative learning rates were tested. Initially, both a smaller and a larger value were explored, and since the larger value showed improvements, an even higher learning rate was added to the experiments. All other training parameters were kept constant across runs. To simplify testing, reduce computational cost, and enable a larger number of runs, the number of epochs was limited to 10. This was sufficient to observe general trends, while more promising configurations were later trained for the full duration.
- Learning_rate_options: {1e-06, 1e-05, 3e-05, 5e-05}
- Patience: 5
- Weight Decay: 0.01
- Loss: cross entropy loss
- Batch Size: 32
- Epochs: 10

**Results:** <br>
With a learning rate of 1e-6, training progressed too slowly. The model required several epochs just to learn copying behavior and had not begun to deviate from the input before early stopping occurred in epoch 5. The best penalised BLEU score to that point was still the one reached in the first epoch. <br>
The default rate of 1e-5 provided a good balance, showing steady improvements without signs of overfitting (validation loss) within 10 epochs (penalised BLEU ≈ 8.1860 at epoch 10). <br>
Increasing the rate to 3e-5 accelerated learning and produced higher penalised BLEU scores (≈ 16.9673 at epoch 10), but training became less stable, with overfitting starting around epoch 6. <br>
At 5e-5, training was faster still, but instability, fluctuations and overfitting appeared as early as epoch 4, and results did not clearly improve over the 3e-5 setting up until epoch 10 (≈ 17.9322). <br>
Overall, 1e-5 emerged as the most reliable choice, offering stable improvements with limited risk of overfitting, while 3e-5 may be useful in later stages when faster progress is desired, albeit at the cost of stability.

<h4> Combinations </h4>

**Explanation:** <br>
A more conservative configuration with a learning rate of 1e-5 and batch size of 32 was compared to a more aggressive setup using a learning rate of 3e-5 and batch size of 16, as both had previously shown improvements without becoming overly unstable.

**Implementation:** <br>
The baseline model with the original loss, weight decay in AdamW and early stopping was run with the two different learning rate and batch size combinations. The chosen hyperparameters can be seen in the lists below.

**Results:** <br>
With patience set to 5, two hyperparameter settings were evaluated. Using a learning rate of 1e-5 with a batch size of 32 (run 10969746) produced the most stable training dynamics, although progress was relatively slow. Signs of overfitting (validation loss) appeared around epoch 10, but the best model was obtained at epoch 37, achieving a penalised BLEU score of 21.903, a negative BLEU of 24.399, and a base score of 46.679. Earlier in training, at epoch 10, the penalised BLEU score was only 6.256, illustrating that improvements required extended training. <br>
By contrast, using a learning rate of 3e-5 with a batch size of 16 (run 10956829) led to much faster improvements, with a penalised BLEU score of 17.326 already reached by epoch 10. The best model, taken from epochs 20, achieved a penalised BLEU score of 25.845, with negative a BLEU score of 32.058, and a base score of 41.923. However, this configuration showed stronger fluctuations and clear overfitting of the validation loss from epoch 4 onward. The rise in penalized BLEU was largely driven by gains in the negative score, while the base score dropped noticeably. This indicates that the model was discouraged from copying the input but not always in ways that improved correctness. Allowing longer patience in early stopping might yield further gains, though the trade-off with base score might prove problematic.

Run 10969746 — Learning rate 1e-5, batch size 32
- Loss: cross entropy loss
- Weight Decay: 0.01
- Overfitting: noticeable from around epoch 10, with fluctuations
- Training stopped: epoch 42 (best model from epoch 37)
- Scores at epoch 37:
    - Penalised BLEU: 21.9025
    - Negative BLEU: 24.3992
    - Base Score: 46.6790
- Penalised BLEU after epoch 10: 6.2561

Runs 10956829 — Learning rate 3e-5, batch size 16
- Loss: cross entropy loss
- Weight Decay: 0.01
- Overfitting: from epoch 4 onwards
- Training stopped: epochs 25 (best model from epochs 20)
- Scores at epoch 20:
    - Penalised BLEU: 25.8451
    - Negative BLEU: 32.0577
    - Base Score: 41.9227
- Penalised BLEU after epoch 10: 17.3258


</details>

<h3> 2. Penalising Input Similarity </h3>

To address the main issue observed during base model training—namely, that the model often copied the input directly—I introduced an additional penalty term to the loss function. This term penalizes high similarity between the input and the generated output, discouraging the model from relying on simple copying.

<details> 

**Explanation:** <br>
I observed that the model could achieve relatively high scores simply by copying the input sentence. This copying behaviour—effectively removing the desired paraphrase's location and type from the input while leaving the actual sentence unchanged — does not reflect genuine paraphrasing. Since paraphrasing involves altering the wording of a sentence while retaining its semantic meaning, I aimed to encourage the model to deviate more strongly from the input. <br>
To achieve this, I considered penalising similarity between input and output representations. Cosine similarity is a widely used measure of embedding similarity and therefore appeared to be an appropriate choice. Although I did not find prior work that applied this approach in the context of paraphrase generation, PyTorch already provides a suitable implementation in the form of the cosine embedding loss, which directly captures the behavior I intended to enforce.

**Implementation:** <br>
An additional loss component was introduced to penalize similarity between input and output embeddings. Specifically, cosine embedding loss was applied using PyTorch’s nn.CosineEmbeddingLoss() with the target set to -1, which encourages dissimilarity between the two representations.
$$
\text{loss}(x, y) =
\begin{cases}
1 - \cos(x_1, x_2), & \text{if } y = 1 \\
\max(0, \cos(x_1, x_2) - \text{margin}), & \text{if } y = -1
\end{cases}
$$
Above shows the function of the cosine embedding loss given in the PyTorch documentation (https://docs.pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html). <br>

This penalty term was then combined with the model’s original cross-entropy loss, as illustrated in the pseudocode below:
    
```python
penalty = cos_embedding_loss(output_embeds, input_embeds, target=-1) 
original_loss = outputs.loss # given by bart model
penalised_loss = (1-l) original_loss + l * penalty
```

Here, _l_ is a weighting factor that controls the influence of the penalty. The original idea was to reduce _l_ in later training epochs, since many parts of the target sequence legitimately overlap with the input. Lowering the weight prevents the model from being pushed unnecessarily far from the input when copying is appropriate. 
Therefore a scheduler was included to decay _l_. <br>
The first version started with a value of `l_start=0.70 ` and then exponentially decays it towards `l_end=0.10` with a decay rate of 0.95 per step. One step equals one batch calculation in this case.
```python
l_step = l_end + (l_start - l_end) * (0.95 ** step)
```
The original idea was extended using a warm up strategy. For a certain fraction of the total steps, which are still batch based, _l_ gets increased as follows.
```python
l_min + (l_max - l_min) * (step / warmup_batches)
```
After the warmup phase it gets decreased as follows.
```python
l_max - (l_max - l_min) * min(decay_batches / total_decay_batches, 1.0)
```
This approach should make training more robust by avoiding excessive pressure on the model in the early stages, when it has not yet developed paraphrasing capabilities. It allows the model to first learn basic copying behavior before gradually encouraging greater diversity in its outputs. <br>
The total number of batches is roughly estimated based on epochs run on average, as determined by previous testing.

**Results:** <br>
Several training runs were performed with different learning rates and batch sizes, testing the penatly loss with an exponential decay of _l_ based on the batch step. <br>
With a learning rate of 1e-05 and a batch size of 64, training progressed steadily, with slow overfitting beginning after roughly 17 epochs. Performance continued to improve until around epoch 45, where the penalised BLEU score stabilized at around 25.3847. A similar configuration with a batch size of 32 produced nearly identical results, showing stable learning, which starts faster but then slows down and in the end needing longer to reach a comparable final performance. <br>
Increasing the learning rate to 3e-05 while keeping the batch size at 64 accelerated early learning but led to slightly lower final penalised BLEU score of 23.2846. When the batch size was reduced to 32, training plateaued earlier, and improvements after the mid-training stage were minimal. At epoch 17 the penalised BLEU score was around 18.1006, staying under 20 until epoch 31, consistently decreasing to numbers in the range of [18,20). The smallest batch size of 16 produced faster initial gains but resulted in less stable learning overall, with stronger fluctuations, a lower base score, and smaller improvements in the penalty-adjusted metrics. <br>
The tests with a batch size of 128 were first added, because of the good results of the tests with a batch size of 64, but then discarded again, as training was so slow, that even at epoch 47 the penalised BLEU score did only reach 9.7694. These numbers were produced with a learning rate of 1e-05. <br>
Overall, the experiments suggest that lower learning rates with moderate batch sizes yield the most stable and effective training dynamics under the penalty with the batch step based exponential decay of _l_. <br>
Set up for above tests:
- Learning Rate_options: {1e-05, 3e-05}
- Batch Size options: {16, 32, 64, 128}
- Patience: 10
- Weight Decay: 0.01
- Loss: cross entropy loss combined with penalty loss
- _l_: batch step based exponential decay

To study the effect of the penalty warm-up fraction, several runs were conducted with otherwise identical training parameters, as can be seen below. <br>
Using a relatively large fraction (0.30) led to slower early improvements, with the model taking longer to reach its best penalised BLEU score of 23.5745. Reducing the fraction to 0.15 improved results, producing both higher overall performance and faster convergence. When the number of total steps was increased (by increasing the epochs for the estimation of the total batch steps from 50 to 70), the improvements appeared even earlier, showing that this setting possibly benefited from slightly longer decay. The two runs produced penalised BLEU scores of 24.9044 and 24.8305 in epoch 54 and 37 respectively. <br>
Testing an even smaller fraction (0.10) resulted in slightly worse performance. While the penalized score improved up until 24.7391, the base BLEU score decreased more noticeably, and the model required more epochs before reaching its best checkpoint, despite also having a higher number of total steps. <br>
Overall, a warm-up fraction of 0.15 with an extended number of total steps offered the best balance between stability, training speed, and score improvements, slightly outperforming both the smaller and larger fractions.

Set up for above tests:
- Learning Rate: 1e-05
- Batch Size: 32
- Patience: 10
- Weight Decay: 0.01
- Loss: cross entropy loss combined with penalty loss
- _l_: warm up strategy

</details>

<h3> 3. Reinforcement Learning using a Generator-Evaluator Setup </h3>
As we can see from the base and penalised BLEU scores and the fact that the model is rewarded for copying, metrics are often flawed. Li et al. (Li et al., 2018) introduced an idea to improve this. They used a model as an evaluator to improve the evaluation of generated paraphrases and, therefore, the feedback given to the model. They proceeded to train a generator model to produce paraphrases and an evaluator model to assess them in a reinforcement learning algorithm.

<details> 

**Explanation:** <br>
Based on the idea introduced by Li et al. (Li et al., 2018) in their paper Paraphrase Generation with Deep Reinforcement Learning, I wanted to use the model trained in the STS task as an evaluator model to act as a metric for my generator, which is the paraphrase type generation bart model. To get a simple baseline the REINFORCE or Vanilla Policy Gradient by Williams (Williams, 1992) was implemented. This could be extended to more complex and better performing algorithms. 

**Implementation:** <br>

The evaluator model outputs a similarity score between [0,5] to evaluate how similar two sentences are. This score is then being rescaled to be between [0,1] to function as a reward. To reduce variance the rewards are being normalised with a baseline reward. The loss is then calculated based on the adjusted rewards and the log probabilities of the sequence. <br>

As the generator was struggling with negative infinity (-inf) values for the probabilities, the softmax function was changed to a more robust version that incorporates clamping to avoid exact zeros.

Originally:

```python 
log_prob = F.log_softmax(step_logits, dim=-1)
```

Changed to: 
```python
probs = F.softmax(step_logits, dim=-1)
probs = probs.clamp(min=1e-12)
log_prob = probs.log()
```
This change was not used in a newer version, but should be saved if in later testing the same issue arises again. <br>
The reward scale is [0,1]. The scale for the logarithmic probabilities for the sequence, are much higher, which is expected for (long) sequences, as they are cumulative across the entire sequence. To handle this difference and prevent a loss in a three-to-four-digits range, the log probabilities were normalised by sequence length to get an average log probability per token.

```python
sequence_log_prob = (log_probs * gen_mask).sum(dim=-1) / sequence_lengths.clamp(min=1)
```

Difficulties with memory when sampling. The function `generator.generate()` could not be used because it internally switches to inference mode and detached the computation graph, which resulted in output probabilities without gradient information. A sampling method was written using the `generator()` function instead. The first version was computationally infeasable, as it took up too much mermory space by storing all logits. The new version only saved the sampled token log probabilities. <br>
In bart_generation_improvement.py the RL is currently commented out and not called. 

**Results:** <br>
The reinforcement learning setup runs on a local machine, but did not run correctly on the compute cluster. The issue appears to be related to tensors not being placed on the correct device, which caused repeated runtime errors. Despite extensive debugging attempts, the exact source of the problem could not be identified. Since my laptop does not have sufficient computational power for full-scale training, I was unable to further debug and test the code. While preliminary tests on the local machine indicated that the model probably was training (batch size of one and a break after the first batch will not lead to much improvement, but more was simply impossible for my laptop) and the rewards were within a reasonable range, the lack of cluster results means that the implementation could not be fully validated. Due to time constraints, I was unable to experiment with a CPU-only setup on the cluster or to isolate the device mismatch more carefully. In the process of troubleshooting, significant parts of the code were rewritten, which may have introduced additional errors that could not be fully resolved before the deadline.

</details>
<h3> 4. Final Run </h3>

<details>

**Explanation:**
The final run was using the setup from section 2. with a penalty loss term using the warm-up strategy. It produced better results than the counter-run using the exponential decay for _l_. 

**Implementation:**
Set up for final run:
- Learning Rate: 1e-05
- Batch Size: 32
- Patience: 10
- Weight Decay: 0.01
- Loss: cross entropy loss combined with penalty loss
- _l_: warm up strategy with frac 0.15

**Results**
It achieved the highest score in epoch 54 with a penalised BLEU score of 25.4342. The base BLEU score is 44.3488 and the negative BLEU score 29.8221. Overfitting with the validation loss still remains a problem, which can also be seen in the plots below.

</details>

<h3> Final Results </h3>

Summary table of all improvements and their respective results.

| Sno. | Description | Penalised Bleu Score Epoch 5 | Best Penalised Bleu Score | Epoch of Best Score |
|------|------------- | --------------------------- | ------------------------------|------------------|
|0| Base Line | 1.982 | - | - |
|1| Weight Decay AdamW | 2.5911 | 21.9025 | 37 |
|1| Optimising Hyperparameters | 13.0276| 23.7526 / 25.8451 (unstable) |  25 |
|2| Penalty loss with batch step based exponential decay for _l_ | 3.2739 | 25.3847 | 45 |
|2| Penalty loss with warmup strategy for _l_ | 4.2235 | 24.8305 | 37 |
|3| Reinforcement Learning | - | - | - |
|4| Final Run | 6.6108 | 25.4342 | 54 |

![alt text](plots/losses_plot_1e-05_32_0.15_WARM_FINAL.png)
![alt text](plots/bleu_plot_1e-05_32_0.15_WARM_FINAL.png)

<h3> Futue Work </h3>

<details> 

**New Data** <br> 
Incorporating new and diverse data would help the model generalise better and improve the imbalance of the different paraphrase types. <br>
However, this presents a challenge, as the existing dataset is the only one incorporating paraphrase types, as far as I know. Annotating enough additional samples to create a new and useful dataset would require too much human labour for this project. This is especially the case given that multiple trained annotators would be needed to actually obtain high-quality data. There are 26 different paraphrase types which are not self-explanatory to the untrained eye. If a sufficiently well-trained model exists, a paraphrase type detection model could be used to automatically annotate existing paraphrases. 
There is a risk of a loss of quality compared to a human expert, especially since a paraphrase type detection model is also limited by the amount of currently available training data. Therefore, as the annotation model might struggle to generalise, this approach would also struggle to produce diverse datasets. Additionally, the new data would be based on existing paraphrase data, which may be unbalanced. <br> 
Another option would be to use a model to generate paraphrase types for synthetic data. However, this also does not seem feasible for the aforementioned reasons. <br>
One solution to these difficulties would be to pre-train the model using a generic paraphrase generation dataset to give it a general understanding of paraphrase generation, and then fine-tune it to recognise different types. This approach could be beneficial in a similar way to how a general understanding of language, obtained through pre-training, aids task-specific downstream language tasks. <br> 

**Overfitting** <br> 
Overfitting can be observed when looking at the validation loss very early on. Future work should focus on mitigating this problem.

**Updated Penalty** <br> 
To mitigate the problem of pushing the output too far from the input, an alternative to weight decay for _l_ could be tested: an updated penalty that calculates the difference between the input-target and input-output distances and uses this as a penalty.
This way the penalty automatically goes towards zero, if the output is as dissimilar to the input as the target. However, this does not guarantee that the difference represents the desired paraphrasing; it only ensures that the margin is the same. To control the correctness of the paraphrasing, the original cross-entropy loss is still required. The ratio between the penalty and the original loss can still be controlled using _l_. <br>
The idea is shown in the following pseudocode. 
```python
diff_pred = cos_embedding_loss(output_embeds, input_embeds, target = -1) 
diff_target = cos_embedding_loss(target_embeds, input_embeds, target = -1) 
penalty = abs(diff_pred - diff_target)
original_loss = outputs.loss # given by bart model
penalised_loss = (1-l) original_loss + l * penalty
```
**Refinement of Reinforcement Algorithm** <br> 
There are more complex reinforcement alogirthms such as the Proximal Policy Optimization (PPO) by Schulman et al. (Schulman et al. 2017). PPO might yield better results, as it can be more stable and robust and uses samples more efficiently (Schulman et al. 2017). <br>
The RL algorithm could also be improved in terms of the evaluator model. As a first step, the model that calculates the similarity score could be improved to output more accurate scores. Currently, the evaluation lacks an understanding of the different types of paraphrase. This could be incorporated into the reward model or an additional, separate model could be used to extend the existing system to include paraphrase type detection in the reward.

**Combining different Approaches** <br>
Originally, I aimed to combine the different approaches in order to evaluate whether their joint application could yield improvements beyond the sum of their individual effects. In particular, I wanted to test whether integrating cross-entropy loss with the penalty term and reinforcement learning would enhance paraphrase generation. To this end, I planned to run the reinforcement learning loop and the supervised training loop sequentially, in both possible orders, to assess whether the training sequence itself had an impact on performance. Unfortunetely, I did not achieve good enough results with the reinforcement learning to get to this testing stage.

</details> 

<h3> References </h3>

<details> 

**Literature**

- ETPC <br>
Venelin Kovatchev, M. Antònia Martí, and Maria Salamó. 2018. ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA)

- Paraphrase Type Paper <br>
Wahle, J., Gipp, B., & Ruas, T. (2023). Paraphrase Types for Generation and Detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 12148–12164). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.746

- AdamW Weight Decay <br>
Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101. 

- RL for Paraphrase Generation <br>
Zichao Li, Xin Jiang, Lifeng Shang, and Hang Li. (2018) Paraphrase Generation with Deep Reinforcement Learning. arXiv preprint arXiv:1711.00279. https://arxiv.org/abs/1711.00279

- The REINFORCE algorithm <br> 
Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256 (1992). https://doi.org/10.1007/BF00992696

- Proximal Policy Optimisation <br> 
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347. https://arxiv.org/abs/1707.06347


**AI Usage** <br>
An AI usage card (https://ai-cards.org/) has been filled out and can be found in the folder _ai_usage_cards_ under _ai_usage_card_esther_hagenkort_.

</details>


## Grete Cluster Execution

1.  **Setup Environment:**
    ```bash
    sbatch setup_gwdg.sh
    ```
    *Wait for this job to complete (`COMPLETED` state) before proceeding.*

2.  **Run Training:**
    ```bash
    sbatch run_train.sh
    ```

**Note:** The training job depends on the environment created by the setup script. Always run `setup_gwdg.sh` first and ensure it finishes successfully.

## AI-Usage 
AI support such as ChatGPT were used, detailed AI-Usage cards are placed int the ai_usage_cards folder.

## Acknowledgement
The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.