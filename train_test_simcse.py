import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from bert import BertModel
from tqdm import tqdm
import numpy as np
import json
from torch import nn
from scipy.stats import spearmanr
from datasets import SentencePairDataset
import csv
from tokenizer import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import random
import logging
from torch.cuda.amp import autocast, GradScaler

BERT_HIDDEN_SIZE = 768
TQDM_DISABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simcse_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_string(s):
    return s.lower().strip()

def load_sts_data(sts_filename, split="train"):
    """Load only STS data, handling None filenames"""
    sts_data = []
    
    if sts_filename is None:
        logger.info(f"No STS file provided for {split}")
        return sts_data
    
    try:
        if split == "test":
            with open(sts_filename, "r", encoding="utf-8") as fp:
                for record in csv.DictReader(fp):
                    sent_id = record["id"].lower().strip()
                    sts_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            sent_id,
                        )
                    )
        else:
            with open(sts_filename, "r", encoding="utf-8") as fp:
                for record in csv.DictReader(fp):
                    sent_id = record["id"].lower().strip()
                    sts_data.append(
                        (
                            preprocess_string(record["sentence1"]),
                            preprocess_string(record["sentence2"]),
                            float(record["similarity"]),
                            sent_id,
                        )
                    )

        logger.info(f"Loaded {len(sts_data)} {split} examples from {sts_filename}")
    except FileNotFoundError:
        logger.error(f"STS file not found: {sts_filename}")
    except Exception as e:
        logger.error(f"Error loading STS data: {e}")
    
    return sts_data

class ImprovedSimCSEBERT(torch.nn.Module):
    """Enhanced BERT model for SimCSE training with multiple improvements"""
    def __init__(self, model_name="bert-base-uncased", dropout_prob=0.1, 
                 pooling_method="mean", use_projection_head=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling_method = pooling_method
        self.use_projection_head = use_projection_head
        
        # Apply dropout to all BERT modules
        for module in self.bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_prob
        
        # Projection head for better representation learning
        if use_projection_head:
            self.projection_head = nn.Sequential(
                nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE),
                nn.ReLU(),
                nn.Linear(BERT_HIDDEN_SIZE, 256)
            )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        if isinstance(outputs, dict):
            hidden_state = outputs["last_hidden_state"]
        else:
            hidden_state = outputs.last_hidden_state
        
        # Different pooling strategies
        if self.pooling_method == "cls":
            embeddings = hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling_method == "mean":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            sum_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_method == "max":
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            hidden_state = hidden_state * input_mask_expanded
            embeddings = torch.max(hidden_state, dim=1)[0]
        
        if self.use_projection_head:
            embeddings = self.projection_head(embeddings)
        
        return embeddings
    
    def simcse_loss(self, emb1, emb2, temperature=0.05, margin=0.5):
        """Improved SimCSE loss with margin and better normalization"""
        batch_size = emb1.size(0)
        
        # Normalize embeddings
        emb1 = F.normalize(emb1, p=2, dim=1)
        emb2 = F.normalize(emb2, p=2, dim=1)
        
        # Similarity matrix
        similarity_matrix = torch.matmul(emb1, emb2.T) / temperature
        
        # Positive pairs are on diagonal
        labels = torch.arange(batch_size).to(emb1.device)
        
        # Add margin for better separation
        loss1 = F.cross_entropy(similarity_matrix - margin * torch.eye(batch_size).to(emb1.device), labels)
        loss2 = F.cross_entropy(similarity_matrix.T - margin * torch.eye(batch_size).to(emb1.device), labels)
        
        return (loss1 + loss2) / 2
    
    def supervised_simcse_loss(self, emb_p, emb_pos, emb_neg, has_negative, temperature=0.05, margin=0.3):
        """Simple and robust supervised loss implementation"""
        batch_size = emb_p.size(0)
        
        # Normalize embeddings
        emb_p = F.normalize(emb_p, p=2, dim=1)
        emb_pos = F.normalize(emb_pos, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(emb_p * emb_pos, dim=1) / temperature
        
        # In-batch negatives
        neg_sim = torch.matmul(emb_p, emb_pos.T) / temperature
        
        # For examples with hard negatives, replace their self-similarity with hard negative
        valid_neg_mask = has_negative.bool()
        if torch.any(valid_neg_mask):
            emb_neg = F.normalize(emb_neg[valid_neg_mask], p=2, dim=1)
            hard_neg_sim = torch.sum(emb_p[valid_neg_mask] * emb_neg, dim=1) / temperature
            neg_sim[valid_neg_mask, torch.arange(batch_size)[valid_neg_mask]] = hard_neg_sim - margin
        
        # Mask out the diagonal (positive pairs)
        mask = torch.eye(batch_size, dtype=torch.bool, device=emb_p.device)
        neg_sim = neg_sim.masked_fill(mask, -10.0)
        
        # Create logits: positive similarity + all negative similarities
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=emb_p.device)
        
        return F.cross_entropy(logits, labels)

class ContrastiveSNLIDataset(Dataset):
    """Dataset that uses SNLI and MNLI sentences for contrastive learning"""
    def __init__(self, snli_path, mnli_path, args, sample_size=None):
        self.dataset = self._load_sentences(snli_path, mnli_path, sample_size)
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _load_sentences(self, snli_path, mnli_path, sample_size):
        """Extract all unique sentences from both SNLI and MNLI"""
        sentences = set()
        
        # Load from SNLI
        if snli_path and os.path.exists(snli_path):
            logger.info(f"Loading sentences from SNLI data at {snli_path}...")
            sentences.update(self._load_single_dataset(snli_path, sample_size, "snli"))
        
        # Load from MNLI
        if mnli_path and os.path.exists(mnli_path):
            logger.info(f"Loading sentences from MNLI data at {mnli_path}...")
            sentences.update(self._load_single_dataset(mnli_path, sample_size, "mnli"))
        
        # Format: (sentence, dummy_label, dummy_sent_id) for compatibility
        formatted_data = [(sentence, 0, f"nli_{i}") for i, sentence in enumerate(sentences)]
        logger.info(f"Loaded {len(formatted_data)} unique sentences from both datasets")
        return formatted_data
    
    def _load_single_dataset(self, file_path, sample_size, dataset_name):
        """Load sentences from a single NLI dataset"""
        sentences = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading {dataset_name} lines"):
                    item = json.loads(line)
                    
                    # Add both premise and hypothesis
                    if item['sentence1']:
                        sentences.add(item['sentence1'])
                    if item['sentence2']:
                        sentences.add(item['sentence2'])
                    
                    if sample_size and len(sentences) >= sample_size:
                        break
        except Exception as e:
            logger.error(f"Error loading {dataset_name} data: {e}")
        
        return sentences
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def pad_data(self, data):
        """Use the same tokenization approach as SentenceClassificationDataset"""
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]  # Dummy labels
        sent_ids = [x[2] for x in data]  # Dummy sent_ids

        encoding = self.tokenizer(sents, return_tensors="pt", padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding["input_ids"])
        attention_mask = torch.LongTensor(encoding["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sents": sents,
            "sent_ids": sent_ids,
        }

        return batched_data

class SupervisedNLIDataset(Dataset):
    """Dataset for supervised SimCSE using both SNLI and MNLI data with hard negatives"""
    def __init__(self, snli_path, mnli_path, args, sample_size=None):
        self.pairs = []  # (premise, positive, negative) tuples
        self._load_nli_pairs(snli_path, "snli", sample_size)
        self._load_nli_pairs(mnli_path, "mnli", sample_size)
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        logger.info(f"Total loaded {len(self.pairs)} supervised training pairs from both SNLI and MNLI")

    def _load_nli_pairs(self, file_path, dataset_name, sample_size):
        if not os.path.exists(file_path):
            logger.warning(f"Warning: {dataset_name} file not found at {file_path}")
            return
            
        logger.info(f"Loading {dataset_name.upper()} pairs from {file_path}...")
        
        premise_dict = {}  # premise -> {'entailment': [], 'contradiction': []}
        pairs_count = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Reading {dataset_name} lines"):
                    item = json.loads(line)
                    
                    premise = item['sentence1']
                    hypothesis = item['sentence2']
                    label = item['gold_label']
                    
                    if premise not in premise_dict:
                        premise_dict[premise] = {'entailment': [], 'contradiction': []}
                    
                    if label in ['entailment', 'contradiction']:
                        premise_dict[premise][label].append(hypothesis)
                    
                    if sample_size and pairs_count >= sample_size:
                        break
                    pairs_count += 1
                        
        except Exception as e:
            logger.error(f"Error loading {dataset_name} data: {e}")
            return
        
        # Create positive-negative pairs
        for premise, hypotheses in premise_dict.items():
            entailments = hypotheses['entailment']
            contradictions = hypotheses['contradiction']
            
            # For each entailment, use one contradiction as hard negative
            for entailment in entailments:
                if contradictions:  # If we have contradiction pairs
                    # Use a random contradiction as hard negative
                    contradiction = random.choice(contradictions)
                    self.pairs.append((premise, entailment, contradiction, dataset_name))
                else:
                    # Fallback: just positive pair without hard negative
                    self.pairs.append((premise, entailment, None, dataset_name))
        
        logger.info(f"Loaded {len(premise_dict)} premises from {dataset_name}")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        premise, positive, negative, dataset_name = self.pairs[idx]
        return premise, positive, negative, f"pair_{idx}"
    
    def pad_data(self, data):
        premises = [x[0] for x in data]
        positives = [x[1] for x in data]
        negatives = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        # Tokenize all sentences
        encoding_premise = self.tokenizer(premises, return_tensors="pt", padding=True, truncation=True)
        encoding_positive = self.tokenizer(positives, return_tensors="pt", padding=True, truncation=True)
        
        token_ids_p = torch.LongTensor(encoding_premise["input_ids"])
        attention_mask_p = torch.LongTensor(encoding_premise["attention_mask"])
        token_ids_pos = torch.LongTensor(encoding_positive["input_ids"])
        attention_mask_pos = torch.LongTensor(encoding_positive["attention_mask"])
        
        # Handle negatives (some might be None)
        negative_sents = [neg if neg is not None else "" for neg in negatives]
        encoding_negative = self.tokenizer(negative_sents, return_tensors="pt", padding=True, truncation=True)
        token_ids_neg = torch.LongTensor(encoding_negative["input_ids"])
        attention_mask_neg = torch.LongTensor(encoding_negative["attention_mask"])
        
        has_negative = torch.tensor([1 if neg is not None else 0 for neg in negatives])

        return (token_ids_p, attention_mask_p, 
                token_ids_pos, attention_mask_pos,
                token_ids_neg, attention_mask_neg,
                has_negative, sent_ids)

    def collate_fn(self, all_data):
        (token_ids_p, attention_mask_p, 
         token_ids_pos, attention_mask_pos,
         token_ids_neg, attention_mask_neg,
         has_negative, sent_ids) = self.pad_data(all_data)

        batched_data = {
            "token_ids_p": token_ids_p,
            "attention_mask_p": attention_mask_p,
            "token_ids_pos": token_ids_pos,
            "attention_mask_pos": attention_mask_pos,
            "token_ids_neg": token_ids_neg,
            "attention_mask_neg": attention_mask_neg,
            "has_negative": has_negative,
            "sent_ids": sent_ids,
        }

        return batched_data

def evaluate_sts_cosine_spearman(model, sts_dataloader, device):
    """Evaluate using cosine similarity (original SimCSE approach)"""
    sts_y_true = []
    sts_y_pred = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(sts_dataloader, desc="STS Eval", disable=TQDM_DISABLE):
            # Get batch data
            b_ids1 = batch["token_ids_1"].to(device)
            b_mask1 = batch["attention_mask_1"].to(device)
            b_ids2 = batch["token_ids_2"].to(device)
            b_mask2 = batch["attention_mask_2"].to(device)
            b_labels = batch["labels"].cpu().numpy()  # Human scores (0-5)
            
            # Get sentence embeddings
            emb1 = model(b_ids1, b_mask1)
            emb2 = model(b_ids2, b_mask2)
            
            # Normalize and compute cosine similarity (-1 to 1)
            emb1 = F.normalize(emb1, dim=1)
            emb2 = F.normalize(emb2, dim=1)
            cosine_sims = torch.sum(emb1 * emb2, dim=1).cpu().numpy()
            
            sts_y_pred.extend(cosine_sims)
            sts_y_true.extend(b_labels)
    
    # Calculate Spearman correlation between cosine sim and human scores
    spearman_corr = spearmanr(sts_y_pred, sts_y_true).correlation
    return spearman_corr

def train_simcse(args):
    # Initialize model with improved architecture
    model = ImprovedSimCSEBERT(
        dropout_prob=args.dropout_prob,
        pooling_method=args.pooling_method,
        use_projection_head=args.use_projection_head
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load sentences from both SNLI and MNLI for contrastive learning
    snli_file = "data/snli_1.0_train.jsonl"
    mnli_file = "data/multinli_0.9_train.jsonl"
    
    train_dataset = ContrastiveSNLIDataset(
        snli_path=snli_file,
        mnli_path=mnli_file,
        args=args,
        sample_size=args.subset_size if args.small_subset else None
    )
    
    if len(train_dataset) == 0:
        logger.error("No training data available. Exiting.")
        return None, -1
    
    # Load STS dev data
    sts_dev_data = load_sts_data("data/sts-similarity-dev.csv", split="dev")
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(
        sts_dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_dataset.collate_fn
    )

    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduling with warmup
    accumulation_steps = max(1, args.gradient_accumulation)
    total_steps = len(dataloader) * args.epochs // accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    # Mixed precision training
    scaler = GradScaler() if device.type == "cuda" else None
    
    best_sts_corr = -1
    best_model_state = None
    patience = 2
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            if scaler:
                with autocast():
                    emb1 = model(input_ids, attention_mask)
                    emb2 = model(input_ids, attention_mask)
                    loss = model.simcse_loss(emb1, emb2, temperature=args.temperature, margin=args.margin)
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                # CPU training
                emb1 = model(input_ids, attention_mask)
                emb2 = model(input_ids, attention_mask)
                loss = model.simcse_loss(emb1, emb2, temperature=args.temperature, margin=args.margin)
                loss = loss / accumulation_steps
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on STS
        sts_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        # Early stopping and model saving
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            no_improvement_count = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_nli/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            logger.info(f"New best model saved with Spearman: {sts_corr:.4f}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Final evaluation
    final_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
    logger.info(f"Final Dev STS Spearman Correlation: {final_corr:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "models/simcse_nli/final_model.pt")
    logger.info("Final model saved.")
    
    return model, best_sts_corr

def train_supervised_simcse(args):
    # Initialize model with improved architecture
    model = ImprovedSimCSEBERT(
        dropout_prob=args.dropout_prob,
        pooling_method=args.pooling_method,
        use_projection_head=args.use_projection_head
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load both SNLI and MNLI datasets
    snli_file = "data/snli_1.0_train.jsonl"
    mnli_file = "data/multinli_0.9_train.jsonl"
    
    train_dataset = SupervisedNLIDataset(
        snli_path=snli_file,
        mnli_path=mnli_file,
        args=args,
        sample_size=args.subset_size if args.small_subset else None
    )
    
    if len(train_dataset) == 0:
        logger.error("No training data available. Exiting.")
        return None, -1
    
    # Load STS dev data
    sts_dev_data = load_sts_data("data/sts-similarity-dev.csv", split="dev")
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(
        sts_dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_dataset.collate_fn
    )

    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=0.01
    )
    
    accumulation_steps = max(1, args.gradient_accumulation)
    total_steps = len(dataloader) * args.epochs // accumulation_steps
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )
    
    scaler = GradScaler() if device.type == "cuda" else None
    
    best_sts_corr = -1
    best_model_state = None
    patience = 2
    no_improvement_count = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, batch in enumerate(progress_bar):
            input_ids_p = batch['token_ids_p'].to(device)
            attention_mask_p = batch['attention_mask_p'].to(device)
            input_ids_pos = batch['token_ids_pos'].to(device)
            attention_mask_pos = batch['attention_mask_pos'].to(device)
            input_ids_neg = batch['token_ids_neg'].to(device)
            attention_mask_neg = batch['attention_mask_neg'].to(device)
            has_negative = batch['has_negative'].to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    emb_p = model(input_ids_p, attention_mask_p)
                    emb_pos = model(input_ids_pos, attention_mask_pos)
                    emb_neg = model(input_ids_neg, attention_mask_neg) if torch.any(has_negative) else None
                    
                    loss = model.supervised_simcse_loss(
                        emb_p, emb_pos, emb_neg, has_negative, 
                        temperature=args.temperature, margin=args.margin
                    )
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if (step + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                emb_p = model(input_ids_p, attention_mask_p)
                emb_pos = model(input_ids_pos, attention_mask_pos)
                emb_neg = model(input_ids_neg, attention_mask_neg) if torch.any(has_negative) else None
                
                loss = model.supervised_simcse_loss(
                    emb_p, emb_pos, emb_neg, has_negative, 
                    temperature=args.temperature, margin=args.margin
                )
                loss = loss / accumulation_steps
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}", 
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        sts_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
        logger.info(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            no_improvement_count = 0
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_supervised/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            logger.info(f"New best model saved with Spearman: {sts_corr:.4f}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    final_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
    logger.info(f"Final Dev STS Spearman Correlation: {final_corr:.4f}")
    
    torch.save(model.state_dict(), "models/simcse_supervised/final_model.pt")
    return model, best_sts_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--small_subset", action="store_true")
    parser.add_argument("--subset_size", type=int, default=20_000)
    parser.add_argument("--supervised", action="store_true", help="Use supervised SimCSE")
    parser.add_argument("--pooling_method", choices=["cls", "mean", "max"], default="mean")
    parser.add_argument("--use_projection_head", action="store_true")
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--local_files_only", action="store_true", help="Use only local model files")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("models/simcse_nli", exist_ok=True)
    os.makedirs("models/simcse_supervised", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    logger.info(f"Starting training with args: {vars(args)}")
    
    if getattr(args, 'supervised', False):
        train_supervised_simcse(args)
    else:
        train_simcse(args)