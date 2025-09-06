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
import datetime

from torch.optim.lr_scheduler import LambdaLR

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

        self.config = config

        # Load pre-tuned SimCSE model or base BERT
        if config.use_pretrained_simcse:
            print(f"Loading pretrained SimCSE model from: {config.simcse_model_path}")
            
            state_dict = torch.load(config.simcse_model_path, map_location='cpu')
            
            # Show model structure
            print(f"Model contains {len(state_dict)} parameters")
            bert_keys = [k for k in state_dict.keys() if k.startswith('bert.')]
            print(f"Found {len(bert_keys)} BERT parameters")
            
            # Initialize with base BERT
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased",
                local_files_only=config.local_files_only
            )
            
            # Extract BERT weights
            bert_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('bert.'):
                    new_key = key[5:]  # Remove 'bert.' prefix
                    bert_state_dict[new_key] = value
            
            # Load the BERT weights
            if bert_state_dict:
                missing_keys, unexpected_keys = self.bert.load_state_dict(bert_state_dict, strict=False)
                print(f"Successfully loaded BERT weights")
                print(f"Missing keys: {len(missing_keys)}")
                print(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                print("Warning: No BERT weights found in SimCSE model.")
                
        else:
            self.bert = BertModel.from_pretrained(
                "bert-base-uncased",
                local_files_only=config.local_files_only
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
        if config.regressor_type == "simple":
            self.sts_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.sts_regressor = nn.Linear(BERT_HIDDEN_SIZE * 3, 1)

        elif config.regressor_type == "complex":
            self.sts_dropout = nn.Dropout(config.hidden_dropout_prob)
            self.sts_regressor = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE * 3, 512), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(config.hidden_dropout_prob),nn.Linear(256, 1))

        elif config.regressor_type == "sbert":
            self.sts_dropout = nn.Dropout(config.hidden_dropout_prob)

        
        # QQP
        self.paraphrase_classifier = nn.Linear(config.hidden_size, 1)
        self.paraphrase_classifier = nn.Dropout(config.hidden_dropout_prob)

        # Paraphrase type detection
        self.paraphrase_type_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_type_classifier = nn.Sequential(nn.Linear(BERT_HIDDEN_SIZE, 26))
        

    def forward(self, input_ids, attention_mask):
        """Takes a batch of sentences and produces embeddings for them."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.config.forward_type == "pooler":
            return outputs["pooler_output"] 
        
        elif self.config.forward_type == "raw_cls":
            if isinstance(outputs, dict):
                return outputs["last_hidden_state"][:, 0, :]
            else:
                return outputs.last_hidden_state[:, 0, :]
        
        elif self.config.forward_type == "sbert_mean":
            token_embeddings = outputs["last_hidden_state"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.config.forward_type == "simcse_sbert":
            token_embeddings = outputs["last_hidden_state"]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            forward_sbert = sum_embeddings / sum_mask

            if isinstance(outputs, dict):
                forward_simcse = outputs["last_hidden_state"][:, 0, :]
            else:
                forward_simcse = outputs.last_hidden_state[:, 0, :]

            return forward_sbert, forward_simcse
        
    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Since the similarity label is a number in the interval [0,5], your output should be normalized to the interval [0,5];
        it will be handled as a logit by the appropriate loss function.
        Dataset: STS
        """

        if self.config.sts_training_type == "standard":
            # Embeddings for both sentences
            emb1 = self.forward(input_ids_1, attention_mask_1)
            emb2 = self.forward(input_ids_2, attention_mask_2)

            features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
            
            features = self.sts_dropout(features)
            logits = self.sts_regressor(features).squeeze()
            
            # Return raw logits for MSE
            return logits

        elif self.config.sts_training_type == "sbert" or self.config.sts_training_type == "simcse":
            # Embeddings for both sentences
            emb1 = self.forward(input_ids_1, attention_mask_1)
            emb2 = self.forward(input_ids_2, attention_mask_2)
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            
            # Cosine similarity
            cosine_sim = torch.sum(emb1 * emb2, dim=1)
            
            return cosine_sim * 2.5 + 2.5  # Scale to [0, 5]
        
        elif self.config.sts_training_type == "simcse_sbert":
            if self.training:
                # Return both during training
                sbert_pred, simcse_pred = self._get_both_predictions(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                return sbert_pred, simcse_pred
            else:
                # During evaluation use only SBERT
                emb1, _ = self.forward(input_ids_1, attention_mask_1)
                emb2, _ = self.forward(input_ids_2, attention_mask_2)
                emb1 = F.normalize(emb1, p=2, dim=1)
                emb2 = F.normalize(emb2, p=2, dim=1)
                cosine_sim = torch.sum(emb1 * emb2, dim=1)
                return cosine_sim * 2.5 + 2.5
    
    def _get_both_predictions(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Internal method that returns both predictions for simcse_sbert training
        """
        # Use the special forward mode for training
        emb1_sbert, emb1_simcse = self.forward(input_ids_1, attention_mask_1)
        emb2_sbert, emb2_simcse = self.forward(input_ids_2, attention_mask_2)
        
        # Calculate both predictions
        emb1_sbert = F.normalize(emb1_sbert, p=2, dim=1)
        emb2_sbert = F.normalize(emb2_sbert, p=2, dim=1)
        sbert_sim = torch.sum(emb1_sbert * emb2_sbert, dim=1)
        sbert_pred = sbert_sim * 2.5 + 2.5
        
        emb1_simcse = F.normalize(emb1_simcse, p=2, dim=1)
        emb2_simcse = F.normalize(emb2_simcse, p=2, dim=1)
        simcse_sim = torch.sum(emb1_simcse * emb2_simcse, dim=1)
        simcse_pred = simcse_sim * 2.5 + 2.5
        
        return sbert_pred, simcse_pred
            
    
    def get_simcse_embeddings(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Get SimCSE embeddings for contrastive loss training
        Returns embeddings for both augmented versions of each sentence
        """
        # Get two different embeddings for each sentence (via dropout)
        emb1_a = self.forward(input_ids_1, attention_mask_1)
        emb1_b = self.forward(input_ids_1, attention_mask_1)  # Different due to dropout
        emb2_a = self.forward(input_ids_2, attention_mask_2)
        emb2_b = self.forward(input_ids_2, attention_mask_2)  # Different due to dropout
        
        return emb1_a, emb1_b, emb2_a, emb2_b
        
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
        self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2
    ):
        """
        Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        Dataset: Quora
        """

        input_ids = torch.cat([input_ids_1, input_ids_2], dim=1)  
        attention_mask = torch.cat([attention_mask_1, attention_mask_2], dim=1)
        mean_embedding = self.forward(input_ids=input_ids,attention_mask=attention_mask)  
        return self.paraphrase_classifier(mean_embedding).squeeze(-1)
    

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
        cls_embedding = self.forward(input_ids, attention_mask)
        cls_embedding = self.paraphrase_type_dropout(cls_embedding)
        logits = self.paraphrase_type_classifier(cls_embedding)
        
        return logits

def ensure_directory_exists(filepath):
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def save_model(model, optimizer, args, config, filepath):
    ensure_directory_exists(filepath)
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


####################### STS Fine-Tuning improvement ##########################
def simcse_contrastive_loss(emb1, emb2, temperature=0.05):
        """SimCSE contrastive loss"""
        batch_size = emb1.size(0)
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Negative similarities
        neg_sim = torch.matmul(emb1, emb2.T) / temperature
        
        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(emb1.device)
        
        # Cross entropy loss for both directions
        loss1 = F.cross_entropy(neg_sim, labels)
        loss2 = F.cross_entropy(neg_sim.T, labels)

        return (loss1 + loss2) / 2
        
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


################################ BONUS TASK #####################################

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

    # Learn the number of steps for the scheduler
    total_steps = 0
    if (args.task == "sts" or args.task == "multitask"):
        total_steps += len(sts_train_dataloader) * args.epochs
    if (args.task == "etpc" or args.task == "multitask"):
        total_steps += len(etpc_train_dataloader) * args.epochs

    num_warmup_steps = int(total_steps * args.warmup_ratio)

    ## Initialize model
    config = {
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": BERT_HIDDEN_SIZE,
        "data_dir": ".",
        "option": args.option,
        "local_files_only": args.local_files_only,
        "regressor_type": args.regressor_type,
        "forward_type": args.forward_type,
        "sts_training_type": args.sts_training_type,
        "use_pretrained_simcse": args.use_pretrained_simcse,
        "simcse_model_path": args.simcse_model_path,
        "max_batches": args.max_batches,
        "file_path": args.filepath,
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
    else:
        # Default optimizer
        lr = args.lr
        optimizer = AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=total_steps
    )

    best_dev_acc = float("-inf")

    # Save train/dev losses and correlations
    train_correlations = []
    dev_correlations = []
    train_losses = []
    dev_losses = []

    ## Training loop
    for epoch in range(args.epochs):

        model.train()
        train_loss = 0
        dev_loss = 0
        num_batches = 0
        dev_num_batches = 0
        

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
            
            batch_idx = 0
            # Standard
            if config.sts_training_type == "standard":
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
                        scheduler.step()

                    train_loss += loss.item()
                    num_batches += 1

                    if batch_idx >= config.max_batches:    
                        break

            # Sbert
            elif config.sts_training_type == "sbert":
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
                        scheduler.step()

                    train_loss += loss.item()
                    num_batches += 1

                    batch_idx += 1

                    if batch_idx >= config.max_batches:    
                        break
            
            # SimCSE
            elif config.sts_training_type == "simcse":
                for batch in tqdm(
                    sts_train_dataloader,
                    desc=f"train-sts-simcse-{epoch+1:02}",
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

                    # Call the method on your model instance
                    emb1_a, emb1_b, emb2_a, emb2_b = model.get_simcse_embeddings(b_ids1, b_mask1, b_ids2, b_mask2)
                    simcse_loss1 = simcse_contrastive_loss(emb1_a, emb1_b)
                    simcse_loss2 = simcse_contrastive_loss(emb2_a, emb2_b)
                    loss = simcse_loss1 + simcse_loss2
                    
                    if config.option == "finetune":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    train_loss += loss.item()
                    num_batches += 1

                    batch_idx += 1

                    if batch_idx >= config.max_batches:    
                        break
            
            # Combined SimCSE + SBERT
            elif config.sts_training_type == "simcse_sbert":

                for batch in tqdm(
                    sts_train_dataloader,
                    desc=f"train-sts-combined-{epoch+1:02}",
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

                    # Get predictions from model (returns tuple: sbert_pred, simcse_pred)
                    predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    sbert_pred, simcse_pred = predictions

                    # SBERT MSE loss
                    sbert_loss = F.mse_loss(sbert_pred, b_labels.view(-1))

                    # SimCSE loss
                    _, emb1_a = model.forward(b_ids1, b_mask1)
                    _, emb1_b = model.forward(b_ids1, b_mask1)
                    _, emb2_a = model.forward(b_ids2, b_mask2)
                    _, emb2_b = model.forward(b_ids2, b_mask2)

                    simcse_loss1 = simcse_contrastive_loss(emb1_a, emb1_b)
                    simcse_loss2 = simcse_contrastive_loss(emb2_a, emb2_b)
                    simcse_loss = simcse_loss1 + simcse_loss2
                    
                    # Combined Loss
                    loss = args.alpha * simcse_loss + (1 - args.alpha) * sbert_loss
                    
                    if config.option == "finetune":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    train_loss += loss.item()
                    num_batches += 1

                    batch_idx += 1

                    if batch_idx >= config.max_batches:    
                        break

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
                logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(logits, b_labels.float())

                if config.option == "finetune":
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                num_batches += 1
        
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
                    scheduler.step()

                train_loss += loss.item()
                num_batches += 1

        train_loss = train_loss / num_batches

        # model.eval()
        # with torch.no_grad():

        #     # Evaluation on dev set
        #     if config.sts_training_type == "standard":
        #         if args.task == "sts" or args.task == "multitask":
        #             for batch in tqdm(sts_dev_dataloader):
        #                 b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
        #                     batch["token_ids_1"].to(device),
        #                     batch["attention_mask_1"].to(device),
        #                     batch["token_ids_2"].to(device),
        #                     batch["attention_mask_2"].to(device),
        #                     batch["labels"].to(device).float(),
        #                 )
                        
        #                 predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        #                 loss = F.mse_loss(predictions, b_labels.view(-1))
        #                 dev_loss += loss.item()
        #                 dev_num_batches += 1

        #     elif config.sts_training_type == "sbert":
        #         if args.task == "sts" or args.task == "multitask":
        #             for batch in tqdm(sts_dev_dataloader):
        #                 b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
        #                     batch["token_ids_1"].to(device),
        #                     batch["attention_mask_1"].to(device),
        #                     batch["token_ids_2"].to(device),
        #                     batch["attention_mask_2"].to(device),
        #                     batch["labels"].to(device).float(),
        #                 )
        #                 predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        #                 loss = F.mse_loss(predictions, b_labels.view(-1))
        #                 dev_loss += loss.item()
        #                 dev_num_batches += 1
                
            
        #     elif config.sts_training_type == "simcse":
        #         if args.task == "sts" or args.task == "multitask":
        #             for batch in tqdm(sts_dev_dataloader):
        #                 b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
        #                     batch["token_ids_1"].to(device),
        #                     batch["attention_mask_1"].to(device),
        #                     batch["token_ids_2"].to(device),
        #                     batch["attention_mask_2"].to(device),
        #                     batch["labels"].to(device).float(),
        #                 )
        #                 predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        #                 loss = F.mse_loss(predictions, b_labels.view(-1))
        #                 dev_loss += loss.item()
        #                 dev_num_batches += 1
            
        #     elif config.sts_training_type == "simcse_sbert":
        #         if args.task == "sts" or args.task == "multitask":
        #             for batch in tqdm(sts_dev_dataloader):
        #                 b_ids1, b_mask1, b_ids2, b_mask2, b_labels = (
        #                     batch["token_ids_1"].to(device),
        #                     batch["attention_mask_1"].to(device),
        #                     batch["token_ids_2"].to(device),
        #                     batch["attention_mask_2"].to(device),
        #                     batch["labels"].to(device).float(),
        #                 )
                        
        #                 predictions = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        #                 simcse_loss = F.mse_loss(predictions, b_labels.view(-1))

        #                 dev_loss += simcse_loss
        #                 dev_num_batches += 1
        
        
        # dev_loss = dev_loss / dev_num_batches

        quora_train_acc, _, _, sst_train_acc, _, _, sts_train_corr, _, _, etpc_train_acc, _, _ = (
            model_eval_multitask(
                sst_train_dataloader,
                quora_train_dataloader,
                sts_train_dataloader,
                etpc_train_dataloader,
                model=model,
                device=device,
                task=args.task,
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
            "multitask": (0, 0),
        }[args.task]

        # # Store metrics
        # train_correlations.append(sts_train_corr)
        # dev_correlations.append(sts_dev_corr)
        # train_losses.append(train_loss)
        # dev_losses.append(dev_loss)

        print(
            f"Epoch {epoch+1:02} ({args.task}): train loss :: {train_loss:.3f}, train :: {train_acc:.3f}, dev :: {dev_acc:.3f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
    
    # Create metrics directory if it doesn't exist
    os.makedirs("metrics", exist_ok=True)

    # Save metrics with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics/sts_metrics_{args.task}_{timestamp}.npz"

    metrics = {
        'train_correlations': np.array(train_correlations),
        'dev_correlations': np.array(dev_correlations),
        'train_losses': np.array(train_losses),
        'dev_losses': np.array(dev_losses)
    }

    np.savez(metrics_filename, **metrics)
    print(f"Saved metrics to {metrics_filename}")


    if args.task == "sts":
        return best_dev_acc
    
    
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

    # Which model to load
    parser.add_argument("--use_pretrained_simcse", action="store_true", 
                       help="Use pre-trained SimCSE model instead of base BERT")
    parser.add_argument("--simcse_model_path", type=str, default="models/simcse_supervised/best_model_epoch3_corr0.7210.pt",
                       help="Path to your pre-trained SimCSE model")
    
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

    # NEW: Regressor type agument
    parser.add_argument(
        "--regressor_type",
        type=str,
        help="Type of regressor to use: simple or complex",
        choices=("simple", "complex", "sbert"),
        default="simple",
    )

    # NEW: Add forward function type argument
    parser.add_argument(
        "--forward_type",
        type=str,
        help="Type of forward function: pooler or raw_cls",
        choices=("pooler", "raw_cls", "sbert_mean", "simcse_sbert"),
        default="raw_cls",
    )

    # NEW: Add STS training type argument
    parser.add_argument(
        "--sts_training_type",
        type=str,
        help="Type of STS training",
        choices=("standard", "sbert", "simcse", "simcse_sbert"),
        default="pooler",
    )
    # NEW: Save only correlation values
    parser.add_argument("--save_results_only", action="store_true",
                       help="Save only results to text file instead of full model")

    # NEW: Add alpha value
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for SimCSE loss in combined training")

    # NEW: Max Batches
    parser.add_argument("--max_batches", type=float, default=180, help="Number of batches tro train on (for STS task only)")

    # NEW: Add warmup ratio argument
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Percentage of total steps for warmup (0.1 = 10%)")

    # Hyperparameters - MOVE THESE UP before the parse_known_args() call
    parser.add_argument(
        "--batch_size", help="sst: 64 can fit a 12GB GPU", type=int, default=64
    )
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
        default=1e-5,  # Set a default, we'll update it later
    )
    parser.add_argument("--local_files_only", action="store_true")

    # Parse known args to get the option for conditional defaults
    args, _ = parser.parse_known_args()
    
    # Update lr default based on option
    if args.option == "pretrain":
        parser.set_defaults(lr=1e-3)
    else:
        parser.set_defaults(lr=1e-5)

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

    # Filepath argument - add this at the end
    parser.add_argument(
        "--filepath", 
        type=str, 
        default=None,  # Set to None initially
        help="Path to save/load the model"
    )

    # Parse all arguments
    args = parser.parse_args()
    
    # # Now calculate default_filepath using the parsed values
    # if args.filepath is None:
    #     args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"
    
    return args

def main():
    args = get_args()
    
    seed_everything(args.seed)
    
    # Run training and get the final correlation for STS task
    if args.task == "sts":
        correlation = train_multitask(args)
    else:
        train_multitask(args)  # For other tasks, just train without return value
    
    # If we're doing a parameter sweep, save results instead of model
    if hasattr(args, 'save_results_only') and args.save_results_only and args.task == "sts":
        # Create results directory
        os.makedirs("sts_sweep_results", exist_ok=True)
        
        # Create descriptive filename with task type
        filename_parts = [args.sts_training_type]
        filename_parts.append(f"seed_{args.seed}")
        
        if hasattr(args, 'alpha') and args.alpha is not None:
            filename_parts.append(f"alpha_{args.alpha}")
        
        if hasattr(args, 'max_batches') and args.max_batches is not None:
            filename_parts.append(f"batch_{args.max_batches}")
        
        filename_parts.append(f"corr_{correlation:.4f}")
        filename = "_".join(filename_parts) + ".txt"
        filepath = os.path.join("sts_sweep_results", filename)
        
        # Save results to text file
        with open(filepath, 'w') as f:
            f.write(f"Task: {args.task}\n")
            f.write(f"Training type: {args.sts_training_type}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Epochs: {args.epochs}\n")
            f.write(f"Learning rate: {args.lr}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            
            if hasattr(args, 'alpha') and args.alpha is not None:
                f.write(f"Alpha: {args.alpha}\n")
            
            if hasattr(args, 'max_batches') and args.max_batches is not None:
                f.write(f"Max batches: {args.max_batches}\n")
            
            f.write(f"Final correlation: {correlation:.4f}\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
        
        print(f"✓ Saved results to {filepath}")
        print(f"✓ Final correlation: {correlation:.4f}")
    else:
        # Normal operation - test the model
        test_model(args)

if __name__ == "__main__":
    main()

# if __name__ == "__main__":
#     args = get_args()

#     if not hasattr(args, 'filepath') or args.filepath is None:
#         args.filepath = f"models/{args.option}-{args.epochs}-{args.lr}-{args.task}.pt"

#     seed_everything(args.seed)  # fix the seed for reproducibility
#     train_multitask(args)
#     test_model(args)