1. Look closer at class distribution
	- Number of samples
	- Covariance ???

2. Implement EarlyStopping to stop training when Model is overfitting

3. Run Baseline model:
	- batch size: 64
	- lr = 1e-5
	- patience: 5
	- precision: 0.783
	- recall: 0.560
	- f1: 0.904
	- Correlation: 0.134

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

5.1 Varing alphas (give rare classes more weight), with stable gamme (0 -> no effect) :
	- inverse class frequency (0.011): 1 / num
	- square root inverse class frequency (0.109): 1 / sqrt2(num)
	- root inverse class frequency (0.13): 1 / sqrt3(num)
	- root inverse class frequency (0.148): 1 / sqrt4(num)
	- root inverse class frequency (0.136): 1 / sqrt5(num)

5.2 Varing gamma with the optimal alpha (sqrt4)
	- gamma 1 is better than gamma 2
	- both are worse than gamma 0

7. Test over/undersampling 
	- idea pick samples based on inverse class frequency -> but limit how big a class can be at most ... 

