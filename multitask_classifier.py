import argparse
import os
from pprint import pformat
import random
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from datasets import (
    SentenceClassificationDataset,
    SentencePairDataset,
    load_multitask_data,
)
from evaluation import model_eval_multitask, test_model_multitask
from optimizer import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

TQDM_DISABLE = False


# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    """
    This module should use BERT for these tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    (- Paraphrase type detection (predict_paraphrase_types))
    """

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=config.local_files_only
        )

        # Freeze BERT parameters in pretrain mode
        for param in self.bert.parameters():
            if config.option == "pretrain":
                param.requires_grad = False
            elif config.option == "finetune":
                param.requires_grad = True
        
        # Set dropout for BERT
        # SST Classification Head
        # HS: Adding a linear layer for sentiment prediction. Will put this at end of last BERT block.
        # The final BERT embedding is the hidden state of [CLS] token which I will get 
        # as dict['pooler_output'] from output of BertModel.forward().
        self.sentiment_classifier = nn.Linear(BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES) # 768 -> 5
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)

        # STS Regression Head
        self.sts_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sts_regressor = nn.Linear(BERT_HIDDEN_SIZE * 3, 1)

        # QQP - Verbesserter Classifier
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_classifier = nn.Sequential(
            nn.Linear(BERT_HIDDEN_SIZE * 3, 768),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(768, 1)
        )
        # Initialisierung
        nn.init.xavier_uniform_(self.paraphrase_classifier[0].weight)
        nn.init.constant_(self.paraphrase_classifier[0].bias, 0.0)
        nn.init.xavier_uniform_(self.paraphrase_classifier[3].weight)
        nn.init.constant_(self.paraphrase_classifier[3].bias, 0.0)



        # Paraphrase type detection
        self.paraphrase_type_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_type_classifier = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, 26))

        
           
    def mean_pooling(self, model_output, attention_mask):
        """
        Performs mean pooling on the token embeddings, taking into account the attention mask.
        model_output: last_hidden_state from BERT
        attention_mask: attention mask for the input
        Returns: pooled sentence embedding of size [batch_size, hidden_size]
        """
        token_embeddings = model_output["last_hidden_state"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""

        # The final BERT embedding is the hidden state of [CLS] token (the first token).
        # See BertModel.forward() for more details.
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if args.task == "qqp":
            embeddings = self.mean_pooling(outputs, attention_mask)
            embeddings = nn.functional.layer_norm(embeddings, normalized_shape=embeddings.size()[1:])
            return embeddings
        
        return outputs["pooler_output"] 
    
    def forward_etpc(self, input_ids, attention_mask):
        """Use last hidden state for the etpc datase."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs["last_hidden_state"]
        cls_embedding = last_hidden_state[:, 0]
        return cls_embedding
    
    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """
        # Embeddings for both sentences
        emb1 = self.forward(input_ids_1, attention_mask_1)
        emb2 = self.forward(input_ids_2, attention_mask_2)

        features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
        
        features = self.sts_dropout(features)
        logits = self.sts_regressor(features).squeeze()
        
        # Return raw logits for MSE
        return logits
        
    def predict_sentiment(self, input_ids, attention_mask):
        """
        Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        Dataset: SST
        """
        ### TODO
        # HS: Get the sequence output from bert's forward pass and then pass it through the to bring it from 768->5 dimensions.
        # The logits will be the output of the sentiment classifier.
        sequence_output = self.forward(input_ids, attention_mask)
        logits = self.sentiment_classifier(sequence_output)
        return logits
        # raise NotImplementedError

    def predict_paraphrase(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, return_embeddings=False
    ):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """
        
        """
        
        SBERT Implementation for paraphrase detection.
        Processes each sentence separately and combines their embeddings for classification.
        """
        # Get embeddings for both sentences
        u = self.forward(input_ids_1, attention_mask_1)
        v = self.forward(input_ids_2, attention_mask_2)

        u = self.paraphrase_dropout(u)
        v = self.paraphrase_dropout(v)

        if return_embeddings:
            # Für MNRL: Gib die Embeddings zurück, um sie im Loss zu vergleichen
            return u, v

        # Nur für Inferenz: Der originale Pfad zur Klassifikation
        abs_diff = torch.abs(u - v)
        combined_features = torch.cat([u, v, abs_diff], dim=-1)
        logits = self.paraphrase_classifier(combined_features).squeeze(-1)
        return logits
    
        # LOSS ZUSATZTERM (1) - 10491466
        # MNRL Loss - 10491549

    def predict_paraphrase_types(
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs logits for detecting the paraphrase types.
        There are 26 different types of paraphrases.
        Thus, your output should contain 26 unnormalized logits for each sentence. It will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: ETPC
        """
        # Concatenate the two sentences while avoiding duplicate special tokens
        input_ids = torch.cat([input_ids_1[:, :-1], input_ids_2[:, 1:]], dim=1)
        attention_mask = torch.cat([attention_mask_1[:, :-1], attention_mask_2[:, 1:]], dim=1)

        # Pass the concatenated input through the model to get embeddings
        cls_embedding = self.forward_etpc(input_ids, attention_mask)
        cls_embedding = self.paraphrase_type_dropout(cls_embedding)
        logits = self.paraphrase_type_classifier(cls_embedding)
        
        return logits
    

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"Saving the model to {filepath}.")

## BONUS TASK
def evaluate_etpc_f1(model, dataloader, device):
    """
    Evaluates the model on the ETPC dataset and computes F1 scores
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            b_ids1 = batch['token_ids_1'].to(device)
            b_mask1 = batch['attention_mask_1'].to(device)
            b_ids2 = batch['token_ids_2'].to(device)
            b_mask2 = batch['attention_mask_2'].to(device)
            b_labels = batch['labels'].float().cpu().numpy()

            logits = model.predict_paraphrase_types(b_ids1, b_mask1, b_ids2, b_mask2)
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

            all_preds.append(preds)
            all_labels.append(b_labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    return macro_f1, micro_f1

def train_multitask(args):
    os.makedirs("models", exist_ok=True)

    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    # Load data
    sst_train_data, _, quora_train_data, sts_train_data, etpc_train_data = (
        load_multitask_data(
            args.sst_train,
            args.quora_train,
            args.sts_train,
            args.etpc_train,
            split="train",
        )
    )
    sst_dev_data, _, quora_dev_data, sts_dev_data, etpc_dev_data = load_multitask_data(
        args.sst_dev, args.quora_dev, args.sts_dev, args.etpc_dev, split="train"
    )

    # Initialize dataloaders
    sst_train_dataloader = None
    sst_dev_dataloader = None
    quora_train_dataloader = None
    quora_dev_dataloader = None
    sts_train_dataloader = None
    sts_dev_dataloader = None
    etpc_train_dataloader = None
    etpc_dev_dataloader = None

    # SST dataset
    if args.task == "sst" or args.task == "multitask":
        sst_train_data = SentenceClassificationDataset(sst_train_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_train_dataloader = DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
        )
    # QQP dataset
    elif args.task == "qqp" or args.task == "multitask":
        qqp_train_data = SentencePairDataset(quora_train_data, args)  # Dataset für Satzpaare
        qqp_dev_data = SentencePairDataset(quora_dev_data, args)

        quora_train_dataloader = DataLoader(
            qqp_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=qqp_train_data.collate_fn,
        )
        quora_dev_dataloader = DataLoader(
            qqp_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=qqp_dev_data.collate_fn,
        )
        
    # STS dataset
    elif args.task == "sts" or args.task == "multitask":
        sts_train_data = SentencePairDataset(sts_train_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args)

        sts_train_dataloader = DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
        )

    ## BONUS TASK
    # ETPC dataset
    elif args.task == "etpc" or args.task == "multitask":

        # Train and dev split as here is no dev dataset given
        train_raw, dev_raw = train_test_split(
            etpc_train_data, 
            test_size=0.2,
            random_state=args.seed
        )

        etpc_train_dataset = SentencePairDataset(train_raw, args)
        etpc_dev_dataset = SentencePairDataset(dev_raw, args)

        etpc_train_dataloader = DataLoader(
            etpc_train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=etpc_train_dataset.collate_fn,
        )
        etpc_dev_dataloader = DataLoader(
            etpc_dev_dataset,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=etpc_dev_dataset.collate_fn,
        )

        ## Print label distribution for ETPC
        # train_labels = np.vstack([np.array(ex[2], dtype=np.float32) for ex in train_raw])
        # dev_labels = np.vstack([np.array(ex[2], dtype=np.float32) for ex in dev_raw])
        # print("ETPC train label distribution (mean per class):", train_labels.mean(axis=0))
        # print("ETPC dev label distribution (mean per class):", dev_labels.mean(axis=0))


    ## Initialize model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
    }
    config = SimpleNamespace(**config)

    separator = "-" * 30
    print(separator)
    print("    BERT Model Configuration")
    print(separator)
    print(pformat({k: v for k, v in vars(args).items() if "csv" not in str(v)}))
    print(separator)

    model = MultitaskBERT(config)
    model = model.to(device)

    etpc_loss = nn.BCEWithLogitsLoss() # Loss for the etpc task (BONUS TASK)

    if args.task == "etpc":
        # Specific parameters for ETPC
        lr = 2e-5
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=0.01,
            correct_bias=False,
        )
    elif args.task == "qqp":
        bert_parameters = []
        classifier_parameters = []

        for name, param in model.named_parameters():
            if "bert" in name: 
                bert_parameters.append(param)
            else: 
                classifier_parameters.append(param)


        optimizer_grouped_parameters = [
            {"params": bert_parameters, "lr": 2e-5, "weight_decay": 0.01},   
            {"params": classifier_parameters, "lr": args.lr,"weight_decay": 0.01}, 
        ]

        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        # Default optimizer
        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)


    best_dev_acc = float("-inf")

    ## Training loop
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        num_batches = 0

        # SST training
        if args.task == "sst" or args.task == "multitask":
            for batch in tqdm(
                sst_train_dataloader,
                desc=f"train-sst-{epoch+1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids, b_mask, b_labels = (
                    batch["token_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device),
                )

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(logits, b_labels.view(-1))

                if config.option == "finetune":
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        # STS training
        if args.task == "sts" or args.task == "multitask":
            for batch in tqdm(
                sts_train_dataloader,
                desc=f"train-sts-{epoch+1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch["token_ids_1"].to(device),
                    batch["attention_mask_1"].to(device),
                    batch["token_ids_2"].to(device),
                    batch["attention_mask_2"].to(device),
                    batch["labels"].to(device).float(),
                )

                optimizer.zero_grad()
                predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                loss = F.mse_loss(predictions, b_labels.view(-1))

                if config.option == "finetune":
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1

        # QQP training
        if args.task == "qqp" or args.task == "multitask":
            for batch in tqdm(
                quora_train_dataloader, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                )

                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)
                
                
                optimizer.zero_grad()

                # 1. Holen Sie die Embeddings für MNRL-style Contrastive Learning
                u, v = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, return_embeddings=True)

                # 2. Contrastive Loss berechnen (wie zuvor)
                u_norm = F.normalize(u, p=2, dim=1)
                v_norm = F.normalize(v, p=2, dim=1)
                similarity_matrix = torch.mm(u_norm, v_norm.T)
                temperature = 0.05
                similarity_matrix = similarity_matrix / temperature
                labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
                contrastive_loss = F.cross_entropy(similarity_matrix, labels)

                # 3. HOLEN SIE AUCH DIE LOGITS FÜR DEN KLASSIKATOR
                logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2, return_embeddings=False)

                # 4. Binary Cross-Entropy Loss für Klassifikation berechnen
                cls_loss = F.binary_cross_entropy_with_logits(logits, b_labels.float())

                # 5. KOMBINATION BEIDER LOSSES (Hyperparameter anpassbar)
                total_loss = cls_loss + 0.1 * contrastive_loss  # Contrastive Loss als Regularizer

                if config.option == "finetune":
                    total_loss.backward()
                    optimizer.step()

                train_loss += total_loss.item()
                
                num_batches += 1
                if num_batches > 10 and args.fastEpoch:
                    break
        
        ## BONUS TASK
        # etpc training
        if args.task == "etpc":
            for batch in tqdm(
                etpc_train_dataloader,
                desc=f"train-etpc-{epoch+1:02}",
                disable=TQDM_DISABLE,
            ):
                b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
                    batch['token_ids_1'].to(device),
                    batch['attention_mask_1'].to(device),
                    batch['token_ids_2'].to(device),
                    batch['attention_mask_2'].to(device),
                    batch['labels'].to(device).float(),
                )

                optimizer.zero_grad()
                logits = model.predict_paraphrase_types(b_ids1, b_mask1, b_ids2, b_mask2) # Orientation on the ETPC evaluation evaluation.py
                    
                loss = etpc_loss(logits, b_labels)

                if config.option == "finetune":
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1
        

        train_loss = train_loss / num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
                fast=args.fastEpoch,
            )
        )

        quora_dev_acc, _, _, sst_dev_acc, _, _, sts_dev_corr, _, _, etpc_dev_acc, _, _ = (
            model_eval_multitask(
                sst_dev_dataloader,
                quora_dev_dataloader,
                sts_dev_dataloader,
                etpc_dev_dataloader,
                model=model,
                device=device,
                task=args.task,
                fast=args.fastEpoch,
            )
        )

        ## BONUS TASK
        if args.task == "etpc":
            macro_f1, micro_f1 = evaluate_etpc_f1(model, etpc_dev_dataloader, device)
            print(f"ETPC Dev Macro F1: {macro_f1:.3f}, Micro F1: {micro_f1:.3f}")

        train_acc, dev_acc = {
            "sst": (sst_train_acc, sst_dev_acc),
            "sts": (sts_train_corr, sts_dev_corr),
            "qqp": (quora_train_acc, quora_dev_acc),
            "etpc": (etpc_train_acc, etpc_dev_acc),
            "multitask": (0, 0),  # TODO
        }[args.task]

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

def test_model(args):
    with torch.no_grad():
        device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
        saved = torch.load(args.filepath)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        return test_model_multitask(args, model, device)

def get_args():
    parser = argparse.ArgumentParser()
    # Training task
    parser.add_argument(
        "--task",
        type=str,
        help='choose between "sst","sts","qqp","etpc","multitask" to train for different tasks ',
        choices=("sst", "sts", "qqp", "etpc", "multitask"),
        default="sst",
    )

    # Model configuration
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)  
    parser.add_argument(
        "--option",
        type=str,
        help="pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated",
        choices=("pretrain", "finetune"),
        default="pretrain",
    )
    parser.add_argument("--use_gpu", action="store_true")

    args, _ = parser.parse_known_args()
    print(f"args: {args}")
    # Dataset paths
    parser.add_argument("--sst_train", type=str, default="data/sst-sentiment-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/sst-sentiment-dev.csv")
    parser.add_argument(
        "--sst_test", type=str, default="data/sst-sentiment-test-student.csv"
    )

    parser.add_argument(
        "--quora_train", type=str, default="data/quora-paraphrase-train.csv"
    )
    parser.add_argument(
        "--quora_dev", type=str, default="data/quora-paraphrase-dev.csv"
    )
    parser.add_argument(
        "--quora_test", type=str, default="data/quora-paraphrase-test-student.csv"
    )

    parser.add_argument(
        "--sts_train", type=str, default="data/sts-similarity-train.csv"
    )
    parser.add_argument("--sts_dev", type=str, default="data/sts-similarity-dev.csv")
    parser.add_argument(
        "--sts_test", type=str, default="data/sts-similarity-test-student.csv"
    )

    # TODO
    # You should split the train data into a train and dev set first and change the
    # default path of the --etpc_dev argument to your dev set.
    parser.add_argument(
        "--etpc_train", type=str, default="data/etpc-paraphrase-train.csv"
    )
    parser.add_argument("--etpc_dev", type=str, default="data/etpc-paraphrase-dev.csv")
    parser.add_argument(
        "--etpc_test",
        type=str,
        default="data/etpc-paraphrase-detection-test-student.csv",
    )

    # Output paths
    parser.add_argument(
        "--sst_dev_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sst_test_out",
        type=str,
        default=(
            "predictions/bert/sst-sentiment-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sst-sentiment-test-output.csv"
        ),
    )

    parser.add_argument(
        "--quora_dev_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--quora_test_out",
        type=str,
        default=(
            "predictions/bert/quora-paraphrase-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/quora-paraphrase-test-output.csv"
        ),
    )

    parser.add_argument(
        "--sts_dev_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--sts_test_out",
        type=str,
        default=(
            "predictions/bert/sts-similarity-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/sts-similarity-test-output.csv"
        ),
    )

    parser.add_argument(
        "--etpc_dev_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-dev-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-dev-output.csv"
        ),
    )
    parser.add_argument(
        "--etpc_test_out",
        type=str,
        default=(
            "predictions/bert/etpc-paraphrase-detection-test-output.csv"
            if not args.task == "multitask"
            else "predictions/bert/multitask/etpc-paraphrase-detection-test-output.csv"
        ),
    )
    parser.add_argument("--fastEpoch", type=bool, default=False)
    # Hyperparameters
    parser.add_argument(
        "--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-3 if args.option == "pretrain" else 1e-5,
    )
    parser.add_argument("--local_files_only", action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = (
        f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"  # save path
    )
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)