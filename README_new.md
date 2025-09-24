
### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types

**Explanation** <br>
Paraphrase Type Generation is a specialisation of the Paraphrase Generation task.    
A paraphrase is a rewording of a sentence without changing the semantic meaning of it.
Wahle et al. (Wahle et al. 2023) introduced a new approach to incorporate paraphrase types into models for paraphrase detection and generation in their paper Paraphrase Types for Generation and Detection. They showed improvements compared to the generic tasks without types. TODO more thourough! <br>
Their dataset (https://huggingface.co/datasets/jpwahle/etpc) is used for this task, which incorporates paraphrase types based on the paper ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negati by Kovatchev et al. (Kovatchev et al., 2018). <br>

**Implementation** <br>
In this task the input is a combination of the input sentence made of a sequence of words $W = [w_1, w_2,...]$, the locations of segments $L = [l_1, l_2,...]$ where the paraphrase should occur and the paraphrase type ids $T = [t_1, t_2, ...]$. This information gets concatenated with a special token to indicate separation and then tokanized by the bart_large tokenizer by facebook with true padding to be ready for the Bart model. % TODO CHECK! and include link to the facebook tokenizer or remove the explicit example
The output should be another sentence made off a sequence of words $O = [o_1, o_2, ...]$ with the same semantic meaning as $W$, but not the same sequence of words.

**Motivation** <br>
- TODO: why important ? -> possible applications ([cite RL paraphrase paper] mention retrival based question answering, semantic parsing, query reformulation in web search and data augmentation for dialog systems as possible applications.) (also check terrys paper!)

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
This approach makes training more robust by avoiding excessive pressure on the model in the early stages, when it has not yet developed paraphrasing capabilities. It allows the model to first learn basic copying behavior before gradually encouraging greater diversity in its outputs. <br>
The total number of batches is estimated based on 50 epochs, as determined by previous testing.

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

</details>

<h3> Reinforcement Learning using a Generator-Evaluator Setup </h3>

Metrics often are BAD -> introducing a model as evaluator to improve the evaluation of generated paraphresed and therefore the feedback given to the model.

For this a reinforcement learning set-up was used, introducing the BLUB model as an evalutor to act as a metric for the generator, which is the bart model. 

It's giving a similarity score to evaluate how similar two sentences are. As using the other model as an evaluator means I cannot update "normally" I need to use reinforcement learning. -> check for copying

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

| Sno. | Description | Penalised Bleu Score Epoch 5 | Best Penalised Bleu Score | Epoch of Best Score |
|------|------------- | --------------------------- | ------------------------------|------------------|
|0| Base Line | 1.982 | - | - |
|1| Weight Decay AdamW | 2.5911 | 21.9025 | 37 |
|1| Optimising Hyperparameters | 13.0276| 23.7526 / 25.8451 (unstable) |  25 |
|2| Penalty loss with batch step based exponential decay for _l_ | 3.2739 | 25.3847 | 45 |
|2| Penalty loss with warmup strategy for _l_ |  |  |  |

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

</details> 

<h3> References </h3>

<details> 

**Literature**

- ETPC <br>
Venelin Kovatchev, M. Antònia Martí, and Maria Salamó. 2018. ETPC - A Paraphrase Identification Corpus Annotated with Extended Paraphrase Typology and Negation. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA)

- Paraphrase Type Paper <br>
Wahle, J., Gipp, B., & Ruas, T. (2023). Paraphrase Types for Generation and Detection. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (pp. 12148–12164). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.746

- AdamW Weight Decay<br>
Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. arXiv preprint arXiv:1711.05101

- The REINFORCE algorithm <br> 
Williams, R.J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach Learn 8, 229–256 (1992). https://doi.org/10.1007/BF00992696

- Proximal Policy Optimisation <br> 
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347. https://arxiv.org/abs/1707.06347


**AI Usage** <br>
An AI usage card (https://ai-cards.org/) has been filled out and can be found in the folder _ai_usage_cards_ under _ai_usage_card_esther_hagenkort_.

</details>


