Question -> Explanations -> Implementations -> Results

1. 
	- What causes the high discrepancy between accuracy and Mathews Correlation ?
	- While Accuracy and Matthews Correlation measure different things, it still was somewhat surprising that a model can have a 0.9+ accuray and a ~0 Matthwes Correlation. To investigate this phenomenon we implemented additional metrics (precision, recall, f1) that are more intuitiv to interpret. Additionally we looked at the occurence rates of the label and at the predictions we submitted in the first stage.
	- To implement the metrics we calculated the TP, FP, TN, FN values for all samples and then used the respective formulas.
	- The high Precision (0.806) shows that the majority of the positivly predicted examples are correct, however the low Recall (0.575) suggests that the model fails to detect many samples as positive. The manual viewing of the results and the plotting (cite) of the class distribution reveiled a likely Hypothesis for why this is. The results showed that only a few classes were predicted (almost) all the time. The class distribution showed that these were the classes with the most frequent occurence in the dataset. The dataset in general was very unevenly distributed between the labels, half a dozen had less then 100 samples, the smallest label only having four occurences, while others having 2705 occurences (in a dataset of 2730 samples). This poses a large problem at both ends of the spectrum for the very small classes there is not enough information for the model to get a good feel for "what they are" and for the large classes there are not enough negative samples. 

2. 
	- What is a resonable baseline to be improved ?
	- In the first stage we were give a set of parameters with which we were supposed to train the model. Before running any Experiments we wanted to set a slightly better "baseline" to better be able to judge weather subsequent methods really show an improvement. For this we tested different batch sizes and learning rates and implemented Early Stopping to not have to worry about the correct number of epochs.
	- We implemented a minimal early stopping class which is initialized with a patience values (whow many epochs no improvement is allowed) and a path were the model is stored. Futhermore there is a "call" function, which takes the model and the current validation score as input. If there is an improvement in the validation score the model snapshot is saved otherwise the patience counter is increased. The Class has a early Stopp variable which can be used to check whether the training should be stopped. After these changes we exepected a moderate increase in performance.
	- The parameters remained similar to the ones we used for the previous stage (lr: 1e-5, batchsize: 64), but the usage of early stopping made a large difference. The Mattews Correlation increased from ~0 in the first stage to 0.125. This performance increase was larger than expected and indicates that the training time for stage one was too short.

		batch size: 64
		lr = 1e-5
		patience: 5
		precision: 0.806
		recall: 0.575
		f1: 0.671
		Correlation: 0.125

3.
	- Is the model not complex enough to solve the problem ?
	- The model model from the first stage had a single linear layer. First we wanted to check if additional linear layers might increase performance. The idea was to give the model might only classify examples along the "most valuable" boundaries since it only has the ability to use a linear boundary. And that thus a model that can differenciate along more complex boundaries might perform better.
	- In the implementation we replaced the single layer with 2 (768 -> 64 -> 26) or 3 (768 -> 128 -> 64 -> 26) layers respectivly. Between the layers Relu and Dropout (0.2) was used execept in the last layer where the sigmoid function remained as before. We expected minimal results at best.
	- The result for were much worse than the baseline, with the three layer version (0.022) beeing better than the two layer one (0.009). 

4. 
	- Do the ideas behind Focal Loss help increasing the performance in this task.
	- To try to adress the shortcomings of the dataset, we used an idea from computer vision called Focal Loss. It is a Loss function which uses 2 separate ideas to address class imbalance in dense object detection tasks. The alpha parameter tries to mitigate the class inbalance, by weightening the loss in favor of rarely occuring classes. The strength of this weightning is determined by the alpha parameter/s. The gamma parameter tries to focuss on "hard" samples, instead of beeing sidetracked by the easy ones. Both should work in our favour, we identified the class inbalance as the major problem in this task, so if the weightening works well, it might improve the performance significantly. The gamma paramter is more of a wildcard, but the large labels are predicted with high confidence so averting the loss away from those would also be good.
	- We implemented the Focal Loss

5. Does over/undersampling help mitigate the class inbalance ?

6. Does a simpler approach to over/undersampling increase performance ?

7. Putting together what works !


3. Does an alpha Balancing Factor (Focal Loss) increase the performance
4. Does a gamma Factor that decreases the Impact of easy samples increase the Performance

5. Does over/undersampling increase the performance of the model

Answers:
1. Very imbalance Class distribution, leads to a "default" answer beeing learned and then repeated for all examples.
2. Additional Linear Layers decrease the performance of the model !






1. Look closer at class distribution
	- Number of samples
	- Covariance ???

2. Implement EarlyStopping to stop training when Model is overfitting

3. Run Baseline model:

4. Check if more complex model performs better
4.1 single layer 64 nodes + 0.2 dropout between 1 and 2
	- batch size: 64
	- lr = 1e-5
	- patience: 5
	- precision: 0.963
	- recall: 0.307
	- f1: 0.466
	- Correlation: 0.009

	-> When the model is predicting samples they are correct, but it misses many samples ...

4.2 single layer 12nodes + 0.2 dropout between 1 and 2
	- batch size: 64
	- lr = 1e-5
	- patience: 5
	- precision: 0.869
	- recall: 0.409
	- f1: 0.627
	- Correlation: 0.022

	-> little better but stiff far away from single layer

5. A review of methods for imbalanced multilabel classification: (Tarekegn 2021)

6. Test Focal Loss

6.1 Varing alphas (give rare classes more weight), with stable gamme (0 -> no effect) :
	- inverse class frequency (0.011): 1 / num
	- square root inverse class frequency (0.109): 1 / sqrt2(num)
	- root inverse class frequency (0.13): 1 / sqrt3(num)
	- root inverse class frequency (0.148): 1 / sqrt4(num)
	- root inverse class frequency (0.136): 1 / sqrt5(num)

6.2 Varing gamma with the optimal alpha (sqrt4)
	- gamma 1 is better than gamma 2
	- both are worse than gamma 0

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
