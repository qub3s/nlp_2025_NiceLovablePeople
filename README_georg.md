Question -> Explanations -> Implementations -> Results

Questions:
1. What causes the high accuracy, but low performance in the Mathews Correlation.
- Implement basic NN stuff ...
2. Will a more complex model perform better than a model "minimal" model with a single Linear Layer.
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
	- batch size: 64
	- lr = 1e-5
	- patience: 5
	- precision: 0.806
	- recall: 0.575
	- f1: 0.671
	- Correlation: 0.125

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

4.2 single layer 64 nodes + 0.2 dropout between 1 and 2
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
