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
- bonus: Paraphrase Type Detection with Bert (PTD-Bert) - Identifying paraphrase types and relationships

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

- Accuracies Epoch 6: `train :: 0.896`, `dev :: 0.521`
- Best dev Accuracy: `train :: 0.704`, `dev :: 0.540` (Achieved in epoch 4)
The model finished training in the 6th epoch with a dev accuracy within 2 standard deviations of the baseline. The dev accuracy decreased after epoch 4 hinting at overfitting in training. We will tackle the overfitting problem with hyperparameter tuning in phase 2.

**Training results:**

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

**Evaluation structure**
The ETPC train label distribution (mean label value per class): 
[0.138, 0.049, 0.048, 0.127, 0.179, 0.647, 0.109, 0.049, 0.001, 0.005, 0.198, 0.013, 0.041, 0.006, 0.017, 0.011, 0.112, 0.192, 0.018, 0.075, 0.764, 0.207, 0.092, 0.993, 0.164, 0.023]

It indicates that most classes are very rare (mean values close to 0.01-0.05) and a few classes are very common (e.g. class 23 with 0.993). So it follows that we have severe class imbalance, so its hard to learn rare classes and therefore the standard accuracy (fraction of correct labels) might be missleading. Thats why I decided to also display the macro and micro F1.

**Improvements**
For the second part I will do hyperparameter finetuning to counter overfitting and find the best local minima. Also I will try to impliment the "Siamese + interaction" recipe used by DeBErta on ETPC from the GippLab group (Paraphrase Types for Generation and Detection, Wahle et al.) and other improvements from Chapter 7.

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
