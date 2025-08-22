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

BERT_HIDDEN_SIZE = 768
TQDM_DISABLE = False

def preprocess_string(s):
    return s.lower().strip()

def load_sts_data(sts_filename, split="train"):
    """Load only STS data, handling None filenames"""
    sts_data = []
    
    if sts_filename is None:
        print(f"No STS file provided for {split}")
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

        print(f"Loaded {len(sts_data)} {split} examples from {sts_filename}")
    except FileNotFoundError:
        print(f"STS file not found: {sts_filename}")
    except Exception as e:
        print(f"Error loading STS data: {e}")
    
    return sts_data

class SimCSEBERT(torch.nn.Module):
    """Simplified BERT model for SimCSE training - compatible with your tokenization"""
    def __init__(self, model_name="bert-base-uncased", dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        for module in self.bert.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_prob
        
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
                return outputs["last_hidden_state"][:, 0, :]
        else:
            return outputs.last_hidden_state[:, 0, :]
    
    def simcse_loss(self, emb1, emb2, temperature=0.05):
        """SimCSE contrastive loss"""
        batch_size = emb1.size(0)
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Positive pairs are on the diagonal
        pos_sim = (emb1 * emb2).sum(dim=1) / temperature
        
        # Negative similarities
        neg_sim = torch.matmul(emb1, emb2.T) / temperature
        
        # Create labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size).to(emb1.device)
        
        # Cross entropy loss for both directions
        loss1 = F.cross_entropy(neg_sim, labels)
        loss2 = F.cross_entropy(neg_sim.T, labels)
        
        return (loss1 + loss2) / 2
    
    def supervised_simcse_loss(self, emb1, emb2, labels, temperature=0.05):
        """Supervised SimCSE loss with hard negatives"""
        batch_size = emb1.size(0)
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        
        # Similarity matrix
        sim_matrix = torch.matmul(emb1, emb2.T) / temperature
        
        # Create mask for positive pairs (diagonal for entailment pairs)
        # Since we have mixed batch of positives and negatives, we need to handle labels
        pos_mask = labels.unsqueeze(1)  # Positives have label 1
        
        # For positive pairs, we want to maximize similarity
        # For negative pairs, we want to minimize similarity
        
        pos_sim = sim_matrix * pos_mask.float()
        neg_sim = sim_matrix * (1 - pos_mask.float())
        
        # Contrastive loss
        pos_loss = -torch.log(torch.exp(pos_sim) / torch.exp(sim_matrix).sum(dim=1, keepdim=True))
        neg_loss = -torch.log(1 - torch.exp(neg_sim) / torch.exp(sim_matrix).sum(dim=1, keepdim=True))
        
        # Only consider non-zero elements
        pos_loss = pos_loss[pos_mask.bool()].mean() if pos_mask.sum() > 0 else 0
        neg_loss = neg_loss[(1 - pos_mask).bool()].mean() if (1 - pos_mask).sum() > 0 else 0
        
        return pos_loss + neg_loss

class ContrastiveSNLIDataset(Dataset):
    """Dataset that uses SNLI sentences for contrastive learning"""
    def __init__(self, file_path, args, sample_size=None):
        self.dataset = self._load_sentences(file_path, sample_size)
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _load_sentences(self, file_path, sample_size):
        """Extract all unique sentences from SNLI and format for compatibility"""
        sentences = set()
        print(f"Loading sentences from SNLI data at {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading lines"):
                    item = json.loads(line)
                    
                    # Add both premise and hypothesis
                    if item['sentence1']:
                        sentences.add(item['sentence1'])
                    if item['sentence2']:
                        sentences.add(item['sentence2'])
                    
                    if sample_size and len(sentences) >= sample_size:
                        break
        except Exception as e:
            print(f"Error loading SNLI data: {e}")
            return []
        
        # Format: (sentence, dummy_label, dummy_sent_id) for compatibility
        formatted_data = [(sentence, 0, f"snli_{i}") for i, sentence in enumerate(sentences)]
        print(f"Loaded {len(formatted_data)} unique sentences")
        return formatted_data
    
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
    """Dataset for supervised SimCSE using NLI data"""
    def __init__(self, file_path, args, sample_size=None):
        self.entailment_pairs = []  # Positive pairs
        self.contradiction_pairs = []  # Hard negatives
        self._load_nli_pairs(file_path, sample_size)
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Combine all data
        self.all_data = self.entailment_pairs + self.contradiction_pairs

    def _load_nli_pairs(self, file_path, sample_size):
        print(f"Loading NLI pairs from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading NLI lines"):
                    item = json.loads(line)
                    
                    # Only use entailment and contradiction
                    if item['gold_label'] == 'entailment':
                        self.entailment_pairs.append((
                            item['sentence1'], 
                            item['sentence2'], 
                            1,  # label: positive
                            f"ent_{len(self.entailment_pairs)}"
                        ))
                    elif item['gold_label'] == 'contradiction':
                        self.contradiction_pairs.append((
                            item['sentence1'],
                            item['sentence2'], 
                            0,  # label: negative
                            f"contra_{len(self.contradiction_pairs)}"
                        ))
                    
                    if sample_size and len(self.entailment_pairs) >= sample_size // 2:
                        break
                        
        except Exception as e:
            print(f"Error loading NLI data: {e}")
        
        print(f"Loaded {len(self.entailment_pairs)} entailment pairs and {len(self.contradiction_pairs)} contradiction pairs")

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
    
    def pad_data(self, data):
        sents1 = [x[0] for x in data]
        sents2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        # Tokenize both sentences
        encoding1 = self.tokenizer(sents1, return_tensors="pt", padding=True, truncation=True)
        encoding2 = self.tokenizer(sents2, return_tensors="pt", padding=True, truncation=True)
        
        token_ids1 = torch.LongTensor(encoding1["input_ids"])
        attention_mask1 = torch.LongTensor(encoding1["attention_mask"])
        token_ids2 = torch.LongTensor(encoding2["input_ids"])
        attention_mask2 = torch.LongTensor(encoding2["attention_mask"])
        labels = torch.LongTensor(labels)

        return token_ids1, attention_mask1, token_ids2, attention_mask2, labels, sents1, sents2, sent_ids

    def collate_fn(self, all_data):
        token_ids1, attention_mask1, token_ids2, attention_mask2, labels, sents1, sents2, sent_ids = self.pad_data(all_data)

        batched_data = {
            "token_ids_1": token_ids1,
            "attention_mask_1": attention_mask1,
            "token_ids_2": token_ids2,
            "attention_mask_2": attention_mask2,
            "labels": labels,
            "sents_1": sents1,
            "sents_2": sents2,
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
    # Initialize model
    model = SimCSEBERT(dropout_prob=args.dropout_prob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load sentences for contrastive learning
    train_file = "data/snli_1.0_train.jsonl"
    train_dataset = ContrastiveSNLIDataset(
        file_path=train_file,
        args=args,
        sample_size=args.subset_size if args.small_subset else None
    )
    
    if len(train_dataset) == 0:
        print("No training data available. Exiting.")
        return None, -1
    
    # Load STS dev data using your original SentencePairDataset
    sts_dev_data = load_sts_data("data/sts-similarity-dev.csv", split="dev")
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_dev_dataloader = DataLoader(
        sts_dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_dataset.collate_fn
    )

    # Use the dataset's collate_fn for proper tokenization
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_sts_corr = -1
    best_model_state = None
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['token_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Pass the same batch through twice with different dropout
            emb1 = model(input_ids, attention_mask)
            emb2 = model(input_ids, attention_mask)
            
            loss = model.simcse_loss(emb1, emb2, temperature=args.temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on STS after each epoch USING COSINE SIMILARITY
        sts_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
        print(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_nli/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            print(f"New best model saved with Spearman: {sts_corr:.4f}")
    
    # Final evaluation
    final_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
    print(f"Final Dev STS Spearman Correlation: {final_corr:.4f}")
    
    torch.save(model.state_dict(), "models/simcse_nli/final_model.pt")
    print("Final model saved.")
    
    return model, best_sts_corr

def train_supervised_simcse(args):
    # Initialize model
    model = SimCSEBERT(dropout_prob=args.dropout_prob)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load supervised NLI data
    train_file = "data/snli_1.0_train.jsonl"
    train_dataset = SupervisedNLIDataset(
        file_path=train_file,
        args=args,
        sample_size=args.subset_size if args.small_subset else None
    )
    
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
        collate_fn=train_dataset.collate_fn
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    best_sts_corr = -1
    best_model_state = None
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids1 = batch['token_ids_1'].to(device)
            attention_mask1 = batch['attention_mask_1'].to(device)
            input_ids2 = batch['token_ids_2'].to(device)
            attention_mask2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings for both sentences
            emb1 = model(input_ids1, attention_mask1)
            emb2 = model(input_ids2, attention_mask2)
            
            # Use supervised loss
            loss = model.supervised_simcse_loss(emb1, emb2, labels, temperature=args.temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        sts_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
        print(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_supervised/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            print(f"New best model saved with Spearman: {sts_corr:.4f}")
    
    final_corr = evaluate_sts_cosine_spearman(model, sts_dev_dataloader, device)
    print(f"Final Dev STS Spearman Correlation: {final_corr:.4f}")
    
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
    parser.add_argument("--supervised", action="store_true", help="Use supervised SimCSE")  # New flag
    args = parser.parse_args()
    
    os.makedirs("models/simcse_supervised", exist_ok=True)
    
    if args.supervised:
        train_supervised_simcse(args)
    else:
        train_simcse(args)
    
    
