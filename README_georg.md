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

