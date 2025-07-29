# DNLP SS25 Final Project 

  

<div align="left">

<b> NiceLoveablePeople </b> <br/>

Esther Hagenkort: esthako/GOESTERN-1006113 <br/>

Georg Eckardt: (gitname) <br/>

Hamza Ahmed Siddiqui: hamzasiddiqui10 <br/>

Amon Pönitzsch: (gitname) <br/>

Leonardo Christian da Camara Silva: (gitname) <br/>

</div>

  

## Introduction
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

  

To install requirements and all dependencies using conda, run:

  

```sh

conda env create -f environment.yml

```

  

The environment is activated with `conda activate dnlp2`.

Additionally, the POS and NER tags need to be downloaded. This can be done by running `python -m spacy download en_core_web_sm`.

  

Alternatively, use the provided script `setup.sh`.

The script will create a new conda environment called `dnlp2` and install all required packages.

## Implementation & Contribution
We followed the instructions, adapted hyperparameters when needed to avoid overfitting.

Esther Hagenkort: 
- bert.py (revise, comment)
- paraphrase generation

Georg Eckardt: 
- optimizer
- bert.py
- paraphrase type detection

Hamza Ahmed Siddiqui:
- bert.py
- Stanford Sentiment Treebank (SST) - Sentiment analysis

Amon Pönitzsch:

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

- Best dev Accuracy: `0.530`

**Training results:**

### Quora Question Pairs (QQP) - Question similarity
**Hyperparameters:**
- mode: `finetune`

- epochs: `20`

- learning rate: `8e-5`

- optimizer: `AdamW`

- dropout rate: 

- batch size: `64`

- ... 

**Training results:**

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
- mode: `finetune`

- epochs: `20`

- learning rate: `8e-5`

- optimizer: `AdamW`

- dropout rate: 

- batch size: `64`

- ... 

**Training results:**

### Paraphrase Type Generation (PTG) - Generating diverse paraphrase types
**Hyperparameters:**

- mode: `finetune`

- epochs: `20`

- learning rate: `8e-5`

- optimizer: `AdamW`

- dropout rate: 

- batch size: `64`

- ... 
  

## Contributing

  

The project involves the creation of software and documentation to be released under an open source licence.

This license is the Apache License 2.0, which is a permissive licence that allows the use of the software for

commercial purposes. The licence is also compatible with the licences of the libraries used in the project.

  

To contribute to the project, please follow the following steps:

  

Clone the repository to your local machine.

  

````sh

git clone git@gitlab.gwdg.de:deep-learning-nlp/token-tricksters.git

````

  

Add the upstream repository as a remote and disable pushing to it. This allows you to pull from the upstream repository

but not push to it.

  

````sh

git remote add upstream https://github.com/truas/minbert-default-final-project

git remote set-url --push upstream DISABLE

````

  

If you want to pull from the upstream repository you can use the following commands.

  

````sh

git fetch upstream

git merge upstream/main

````

  

### Pre-Commit Hooks

  

The code quality is checked with pre-commit hooks. To install the pre-commit hooks run the following command.

This is used to ensure that the code quality is consistent and that the code is formatted uniformly.

  

````sh

pip install pre-commit

pre-commit install

````

  

This will install the pre-commit hooks in your local repository. The pre-commit hooks will run automatically before each

commit. If the hooks fail the commit will be aborted. You can skip the pre-commit hooks by adding the `--no-verify` flag

to your commit command.

  

The installed pre-commit hooks are:

  

- [`black`](https://github.com/psf/black) - Code formatter (Line length 100)

- [`flake8`](https://github.com/PyCQA/flake8) Code linter (Selected rules)

- [`isort`](https://github.com/PyCQA/isort) - Import sorter

  

### Grete Cluster

  

To run the multitask classifier on the Grete cluster you can use the `run_train.sh` script. You can change the

parameters in the script to your liking. To submit the script use

  

````sh

sbatch run_train.sh

````

  

To check on your job you can use the following command

  

```sh

squeue --me

```

  

The logs of your job will be saved in the `logdir` directory. The best model will be saved in the `models` directory.

  

To run tensorboard on the Grete cluster you can use the following commands to create a tunnel to your local machine and

start tensorboard.

  

````sh

ssh -L localhost:16006:localhost:6006 <username>@glogin.hlrn.de

module load anaconda3

conda activate dnlp2

tensorboard --logdir logdir

````

  

If you want to run the model on the Grete cluster interactively you can use the following command, which will give you

access to a GPU node with an A100 GPU. This is for testing purposes only and should not be used for training.

  

````sh

srun -p grete:shared --pty -G A100:1 --interactive bash

````

  

## AI-Usage Card

  

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