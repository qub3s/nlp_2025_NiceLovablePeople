import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from bert import BertModel  # your existing BERT import
from tqdm import tqdm
import numpy as np
import json
from torch import nn
from scipy.stats import spearmanr
from datasets import SentencePairDataset
import csv

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
    """Simplified BERT model for SimCSE training"""
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
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
        
        sim_matrix = torch.matmul(emb1, emb2.T) / temperature
        labels = torch.arange(batch_size).to(emb1.device)
        
        return F.cross_entropy(sim_matrix, labels)

class STSEvaluator:
    """Uses your existing STS prediction logic"""
    def __init__(self, simcse_model):
        self.simcse_model = simcse_model
        # Add your regression head for evaluation only
        self.sts_regressor = nn.Linear(BERT_HIDDEN_SIZE * 3, 1)
        self.sts_dropout = nn.Dropout(0.3)
    
    def to(self, device):
        """Move all components to the specified device"""
        self.simcse_model.to(device)
        self.sts_regressor.to(device)
        self.sts_dropout.to(device)
        return self
    
    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """Your existing STS prediction function"""
        emb1 = self.simcse_model(input_ids_1, attention_mask_1)
        emb2 = self.simcse_model(input_ids_2, attention_mask_2)
        
        features = torch.cat([emb1, emb2, torch.abs(emb1 - emb2)], dim=1)
        features = self.sts_dropout(features)
        logits = self.sts_regressor(features).squeeze()
        
        return logits

class SNLIDataset(torch.utils.data.Dataset):
    """Custom dataset loader for local SNLI files"""
    def __init__(self, file_path, tokenizer, max_length=128, sample_size=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_snli_data(file_path, sample_size)
        
    def _load_snli_data(self, file_path, sample_size):
        """Load SNLI data from JSONL files"""
        data = []
        print(f"Loading SNLI data from {file_path}...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Reading lines"):
                    item = json.loads(line)
                    
                    # Skip invalid or poorly formatted examples
                    if (item['gold_label'] not in ['entailment', 'neutral', 'contradiction'] or 
                        not item['sentence1'] or not item['sentence2']):
                        continue
                    
                    data.append({
                        'premise': item['sentence1'],
                        'hypothesis': item['sentence2'], 
                        'label': item['gold_label']
                    })
                    
                    # Early stop if sampling
                    if sample_size and len(data) >= sample_size:
                        break
        except FileNotFoundError:
            print(f"SNLI file not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error loading SNLI data: {e}")
            return []
        
        print(f"Loaded {len(data)} valid examples")
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenization code remains the same as before
        premise = self.tokenizer(
            item['premise'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        hypothesis = self.tokenizer(
            item['hypothesis'],
            truncation=True,
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'premise_ids': premise['input_ids'].squeeze(),
            'premise_mask': premise['attention_mask'].squeeze(),
            'hypothesis_ids': hypothesis['input_ids'].squeeze(),
            'hypothesis_mask': hypothesis['attention_mask'].squeeze(),
            'label': item['label']
        }

def evaluate_sts_spearman(evaluator, sts_dataloader, device):
        """Use your exact evaluation code format with STSEvaluator"""
        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []

        evaluator.simcse_model.eval()  # Set to eval mode
        with torch.no_grad():
            for batch in tqdm(sts_dataloader, desc="STS Eval", disable=TQDM_DISABLE):
                (b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids) = (
                    batch["token_ids_1"],
                    batch["attention_mask_1"],
                    batch["token_ids_2"],
                    batch["attention_mask_2"],
                    batch["labels"],
                    batch["sent_ids"],
                )

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                # Use STSEvaluator's predict_similarity
                logits = evaluator.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)

        # Use Spearman correlation
        spearman_corr = spearmanr(sts_y_pred, sts_y_true).correlation
        return spearman_corr

def train_simcse(args):
    # Initialize model and tokenizer
    model = SimCSEBERT()
    tokenizer = model.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Add memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load your local SNLI dataset
    train_file = "data/snli_1.0_train.jsonl"
    train_dataset = SNLIDataset(
        file_path=train_file,
        tokenizer=tokenizer,
        sample_size=args.subset_size if args.small_subset else None
    )
    
    # Check if we have any training data
    if len(train_dataset) == 0:
        print("No training data available. Exiting.")
        return None, -1
    
    # Load STS dev data - using our simplified function
    sts_dev_data = load_sts_data("data/sts-similarity-dev.csv", split="dev")
    
    # Create dataset and dataloader
    args.local_files_only = False 
    sts_dev_dataset = SentencePairDataset(sts_dev_data, args)
    sts_dev_dataloader = DataLoader(
        sts_dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=sts_dev_dataset.collate_fn
    )

    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    evaluator = STSEvaluator(model).to(device)
    
    best_sts_corr = -1
    best_model_state = None
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            premise_ids = batch['premise_ids'].to(device)
            premise_mask = batch['premise_mask'].to(device)
            hypothesis_ids = batch['hypothesis_ids'].to(device)
            hypothesis_mask = batch['hypothesis_mask'].to(device)
            
            optimizer.zero_grad()
            
            # Get embeddings for both sentences
            premise_emb = model(premise_ids, premise_mask)
            hypothesis_emb = model(hypothesis_ids, hypothesis_mask)
            
            # SimCSE loss
            loss = model.simcse_loss(premise_emb, hypothesis_emb, temperature=args.temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on STS after each epoch
        sts_corr = evaluate_sts_spearman(evaluator, sts_dev_dataloader, device)
        print(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        # Save best model
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_nli/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            print(f"New best model saved and Spearman: {sts_corr:.4f}")
    
    dev_corr = evaluate_sts_spearman(evaluator, sts_dev_dataloader, device)
    print(f"Dev STS Spearman Correlation: {dev_corr:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "models/simcse_nli/final_model.pt")
    print("Final model saved.")
    
    return model, best_sts_corr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--small_subset", action="store_true", help="Use small subset for testing")
    parser.add_argument("--subset_size", type=int, default=20_000, help="Size of subset for hyperparameter tuning")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("models/simcse_nli", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    train_simcse(args)