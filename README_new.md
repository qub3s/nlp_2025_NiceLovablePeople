
### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types

General instructions and task
The dataset by Wahle et al. (Wahle et al. 2023) is used, which incorporates paraphrase types based on the ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negati by Kovatchev et al. (Kovatchev et al., 2018). In their paper Paraphrase Types for Generation and Detection they introduced a new approach to incorporate paraphrase types into modells for paraphrase detection and generation and showed improvements BLA.

<h3> 1. Setup for Improvements: Hyperparameters and Early Stopping </h3>
In preparation for work on improvements, the baseline training has been optimised. This involved testing different hyperparameters and combinations of these, as well as introducing early stopping based on the new metric, the penalised BLEU score.

<details> 

<h4> Weight Decay </h4>

**Explanation:** <br>
-> check the Paper given in DOKU!

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
Text

**Implementation:** <br>
Text

**Results:** <br>
Text

<h4> Learning Rate </h4>

**Explanation:** <br>
Text

**Implementation:** <br>
Text

**Results:** <br>
Text

<h4> Combinations </h4>

**Explanation:** <br>
Text

**Implementation:** <br>
Text

**Results:** <br>
Text

</details>

<h3> 2. Penalising Input Similarity </h3>
To avoid simply copying the input, which was the biggest issue when training the base model, I introduced a penalty.

<details> 

**Explanation:** <br>
Text

**Implementation:** <br>
Text

**Results:** <br>
Text

</details>

<h3> Reinforcement Learning using a Generator-Evaluator Setup </h3>
Metrics often are BAD -> introducing a model as evaluator to improve the evaluation of generated paraphresed and therefore the feedback given to the model.

<details> 

**Explanation:** <br>

   

**Implementation:** <br>
To get a simple baseline the REINFORCE or Vanilla Policy Gradient by Williams (Williams, 1992) was implemented. This could then be extended to more complex and better performing algorithms. <br>
This means EXPLANATION OF IMPLEMENTATION

As the model was struggling with negative infinity (-inf) values for the probabilities, the softmax function was changed to a more robust version that incorporates clamping to avoid exact zeros.

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

The reward scale is [0,1]. The scale for the Sequence Log Probs, the logarithmic probabilities for the sequence, are much higher, which is expected for (long) sequences, as they are cumulative across the entire sequence. To handle this difference and prevent a loss in a three-to-four-digits range, the Log Probabilities were normalised by sequence length to get an average log probability per token.

```python
sequence_log_prob = (log_probs * gen_mask).sum(dim=-1) / sequence_lengths.clamp(min=1)
```


**Results:** <br>
Text

</details>

<h3> Combining different Approaches </h3>
Let's see if the combination of different improvements are more than the sum of their parts and could further improve the paraphrase generation.

<details> 

**Explanation:** <br>
Text

**Implementation:** <br>
Text

**Results:** <br>
Text

</details>


<h3> Results Table </h3>

Summary table of all improvements and their respective results.

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

**Updated Penalty**    
To mitigate the problem of pushing the output too far from the input, an alternative to weight decay for _l_ could be tested: an updated penalty that calculates the difference between the input-target and input-output distances and uses this as a penalty.
This way the penalty automatically goes towards zero, if the output is as dissimilar to the input as the target. However, this does not guarantee that the difference represents the desired paraphrasing; it only ensures that the margin is the same. To control the correctness of the paraphrasing, the original cross-entropy loss is still required. The ratio between the penalty and the original loss can still be controlled using _l_. <br>
The idea is shown in the following pseudocode. 
```python
diff_pred = cos_embedding_loss(predicted_ids, input_ids, target = -1) 
diff_target = cos_embedding_loss(target_ids, input_ids, target = -1) 
penalty = abs(diff_pred - diff_target)
original_loss = outputs.loss # given by bart model
penalised_loss = (1-l) original_loss + l * penalty
```

**Refinement of Reinforcement Algorithm** <br> 
There are more complex reinforcement alogirthms such as the Proximal Policy Optimization (PPO) by Schulman et al. (Schulman et al. 2017). PPO might yield better results, as it can be more stable and robust and uses samples more efficiently (Schulman et al. 2017).

</details> 

<h3> References </h3>

<details> 

**Literature**

- ETPC <br>
Venelin Kovatchev, M. Antònia Martí, and Maria Salamó. 2018. ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA)

- Paraphrase Type Paper <br>
Wahle, J., Gipp, B., & Ruas, T. (2023). Paraphrase Types for Generation and Detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 12148–12164). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.746

- The REINFORCE algorithm <br> 
Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256 (1992). https://doi.org/10.1007/BF00992696

- Proximal Policy Optimisation <br> 
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347. https://arxiv.org/abs/1707.06347

#### RL
- **Citation**

**AI Usage**
An AI usage card (link) has been filled out and can be found in the folder _ai_usage_cards_ under _AI_Usage_Card_Esther_.

</details>


