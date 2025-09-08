### Paraphrase Type Detection


<h3> 1. What causes the high discrepancy between accuracy and Matthews Correlation Coefficient ? </h3>
<details> 
**Explanation:** <br>
While accuracy and Matthews Correlation Coefficient measure different things, it still was somewhat surprising that a model can have a 90% plus accuracy and a ~0 Matthews Correlation. To investigate this phenomenon we implemented additional metrics (precision, recall, f1) that are more intuitive. Additionally we looked at the occurrence rates of the label and at the predictions we submitted in the first stage.

**Implementation:** <br>
To implement the metrics we calculated the TP, FP, TN, FN values for all samples and then used the respective formulas.

**Results:** <br>
The high Precision (0.806) shows that the majority of the positively predicted examples are correct, however the low Recall (0.575) suggests that the model fails to detect many samples as positive. The manual viewing of the results and the plotting of the class distribution revealed a likely Hypothesis for why this is. The results showed that only a few classes were predicted (almost) all the time. The class distribution showed that these were the classes with the most frequent occurrence in the dataset. The dataset in general was very unevenly distributed between the labels, half a dozen had less than 100 samples, the smallest label only having four occurrences, while others having 2705 occurrences (in a dataset of 2730 samples). This poses a large problem at both ends of the spectrum for the very small classes there is not enough information for the model to get a good feel for "what they are" and for the large classes there are not enough negative samples. 
</details>

<h3> 2. Establishing a Reasonable Baseline for Improvement ? </h3>
<details> 

**Explanation:** <br>
In the first stage we were given a set of parameters with which we were supposed to train the model. Before running any Experiments we wanted to set a slightly better "baseline" to better be able to judge whether subsequent methods really show an improvement. For this we tested different batch sizes and learning rates and implemented Early Stopping to not have to worry about the correct number of epochs.

**Implementation:** <br>
We implemented a minimal early stopping class which is initialized with a patience value (how many epochs no improvement is allowed) and a path where the model is stored. Furthermore there is a "call" function, which takes the model and the current validation score as input. If there is an improvement in the validation score the model snapshot is saved otherwise the patience counter is increased. The Class has an early Stopp variable which can be used to check whether the training should be stopped. After these changes we expected a moderate increase in performance.

**Results:** <br>
The parameters remained similar to the ones we used for the previous stage (lr: 1e-5, batchsize: 64), but the usage of early stopping made a large difference. The Matthews Correlation increased from ~0 in the first stage to 0.125. This performance increase was larger than expected and indicates that the training time for stage one was too short.
</details>

<h3> 3. Is the model not complex enough to solve the problem ? </h3>

<details> 
**Explanation:** <br>
The model model from the first stage had a single linear layer. First we wanted to check if additional linear layers might increase performance. The idea was to give the model might only classify examples along the "most valuable" boundaries since it only has the ability to use a linear boundary. And that thus a model that can differentiate along more complex boundaries might perform better.

**Implementation:** <br>
In the implementation we replaced the single layer with 2 (768 -> 64 -> 26) or 3 (768 -> 128 -> 64 -> 26) layers respectively. Between the layers Relu and Dropout (0.2) was used except in the last layer where the sigmoid function remained as before. We expected minimal results at best.

**Results:** <br>
The result for were much worse than the baseline, with the three layer version (0.022) being better than the two layer one (0.009). Interestingly the 2 layer version showed a higher precision and lower recall than the baseline and the 3 layer model. This indicates that the model probably instantly overfits if you add additional linear layers or potentially is already overfitting. 
</details>

<h3> 4. Do the ideas behind Focal Loss help increasing the performance in this task. </h3>

<details> 
**Explanation:** <br>
To try to adress the shortcomings of the dataset, we used an idea from computer vision called Focal Loss. It is a Loss function which uses 2 separate ideas to address class imbalance in dense object detection tasks. The alpha parameter tries to mitigate the class inbalance, by weightening the loss in favor of rarely occuring classes. The strength of this weightning is determined by the alpha parameter/s. The gamma parameter tries to focuss on "hard" samples, instead of beeing sidetracked by the easy ones. Both should work in our favour, we identified the class inbalance as the major problem in this task, so if the weightening works well, it might improve the performance significantly. The gamma paramter is more of a wildcard, but the large labels are predicted with high confidence so averting the loss away from those would also be good. To test which of these ideas works we tested them individually.

**Implementation:** <br>
We implemented the Focal Loss as a Pytorch Loss (nn.Module), which just implements it's mathematical formulation.

**Results:** <br>
</details>

<h3> 5. Does oversamplig help mitigate the effects of class inbalance ? </h3>

<details> 

**Explanation:** <br>
One of the methods to tackle class imbalances is oversampling. Oversampling is the process of duplicating samples to balance out the class frequencies. We tried a method were we assign a score to every sample based on its labels and then sample from this list based on the scores. The scores were calculated on the inverse frequency ($\frac{1}{X}$) of the labels (or their squareroots $1/^{xth}\sqrt{(X)}$ ).

$value = \sum^0_{26} target_x * freq_x / \sum^0_{26} target_x$

Through this in practice we do not create any copies of the data but during sampling we will pull more data which consists of rare classes. 

**Implementation:**  <br>
We implemented this by writing the class Weight_based_sampler, which inherits from the Pytorch class torch.utils.data.Sampler. It calculates the values like described above, turns them into probabilities and then samples from that distribution using np.random.choice. 

**Results:**

</details>

</details>
<h3> 6. Putting together what works ! </h3>

</details>

**Explanation:** <br>

**Implementation:** <br>

**Results:**


</details>

*All Experiements were run on 1e-5 learning rate, patience 5 and batchsize 64.*

| Experiment | Name | precision | recall | f1 | Correlation | 
|--|--|--|--|--|--|
|2|baseline|0.806|0.575|0.671|0.117|
|3|2_layers|0.963|0.307|0.466|0.009|
|3|3_layers|0.869|0.409|0.627|0.022|
|5|1/x|0.835|0.530|0.649|0.095|

A review of methods for imbalanced multilabel classification: (Tarekegn 2021)


7. Test over/undersampling 
	- idea pick samples based on inverse class frequency -> but limit how big a class can be at most ... 
	- 1/x:
		- precision: 0.791
		- recall: 0.480
		- f1: 0.597
		- Correlation: 0.079
	- 1/sqrt(x):
		- precision: 0.850
		- recall: 0.525
		- f1: 0.649
		- Correlation: 0.1
	- 1/sqrt3(x):
		- precision: 0.836
		- recall: 0.527
		- f1: 0.647
		- Correlation: 0.09
	- 1/sqrt4(x):
		- precision: 0.791
		- recall: 0.571
		- f1: 0.664
		- Correlation: 0.139
	- 1/sqrt5(x):
		- precision: 0.820
		- recall: 0.559
		- f1: 0.665
		- Correlation: 0.104

	- Not clear how the hyperparamter changes the result

8. Duplicating the results:
	- 2 parameters that describe the max duplication rate
		- as a percentage of the largest class
		- as a number of max (on average) duplicates per class

9. putting it all together + parameter finetuning
	- random grid search
