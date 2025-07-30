# DNLP SS25 Final Project 

  

<div align="left">

<b> NiceLoveablePeople </b> <br/>

Esther Hagenkort: esthako/GOESTERN-1006113 <br/>

Georg Eckardt: (gitname) <br/>

Hamza Ahmed Siddiqui: hamzasiddiqui10 <br/>

Amon Pönitzsch: 4m0n <br/>

Leonardo Christian da Camara Silva: (gitname) <br/>

</div>

  

## Introduction
TODO
This repository is our baseline implementation of the project for the Deep Learning for Natural Language Processing class at the University of Göttingen. TODO add class

TODO: AI card

Which parts are included, No bonus task have been commited. Who did what tasks, what baseline percentages have been reached.
A sentence about solutions



A pretrained

BERT ([BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805))

model was used as the basis for our experiments. The model was fine-tuned on the three tasks using a multitask learning

approach. The model was trained on the three tasks simultaneously, with a single shared BERT encoder and three separate

task-specific classifiers.

  

## Requirements
TODO
  

To install requirements and all dependencies using conda, run:

  

```sh

conda env create -f environment.yml

```

  

The environment is activated with `conda activate dnlp2`.

Additionally, the POS and NER tags need to be downloaded. This can be done by running `python -m spacy download en_core_web_sm`.

  

Alternatively, use the provided script `setup.sh`.

The script will create a new conda environment called `dnlp` and install all required packages.

## Implementation & Contribution
We followed the instructions and adapted hyperparameters when needed to avoid overfitting.

Esther Hagenkort: 
- bert.py (revise, comment)
- bonus task: bert etpc paraphrase type detection (debugging)
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
  


## Results Part 1
For the baselines we reached the following results with the described hyperparameters and metrics.

### Stanford Sentiment Treebank (SST) - Sentiment analysis
**Hyperparameters:**
- mode: `finetune`

- epochs: `6`

- learning rate: `1e-05`

- optimizer: `AdamW`

- dropout rate: `0.25`

- batch size: `64`

- Accuracies Epoch 6: `train :: 0.908`, `dev :: 0.513`
- Best dev Accuracy: `train :: 0.774`, `dev :: 0.530` (Achieved in epoch 4)
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
- Dev Accuracy Finetuning: `0.868`

### Semantic Textual Similarity (STS) - Measuring text meaning similarity
**Hyperparameters:**
- mode: `finetune`

- epochs: `20`

- learning rate: `8e-5`

- optimizer: `AdamW`

- dropout rate: 

- batch size: `64`

- ... 

**Training results:**

### Paraphrase Type Detection (PTD) - Identifying paraphrase types and relationships
**Hyperparameters:**
- epochs: `10`

- learning rate: `1e-5`

- optimizer: `AdamW`

- dropout rate: `0.1`

- batch size: `64`

- ... 

**Training results:**

### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types
Also tested 10 epochs and a batch size of 128 with no significant improvements, so the lower values were chosen.

**Hyperparameters:**
- epochs: `5`

- learning rate: `1e-5`

- eps = `1e-8`

- optimizer: `AdamW`

- batch size: `32`


### Grete Cluster

To run the tasks on the Grete cluster we adapted and used the `run_train.sh` script given to us.
  

## AI-Usage Card todo
  

Artificial Intelligence (AI) aided the development of this project. For transparency, we provide our [AI-Usage Card](./AI-Usage-Card.pdf/) at the top. The card is based on [https://ai-cards.org/](https://ai-cards.org/).

  

## Acknowledgement TODO
  

The project description, partial implementation, and scripts were adapted from the default final project for the

Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun,

John Cho, and their (large) team (Thank you!)

  

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon

University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),

created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

  

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)

library ([Apache License 2.0](./LICENSE)).

  

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/)

and [Terry Ruas](https://terryruas.com/).
