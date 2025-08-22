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
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # No need for separate tokenizer - it's handled by the dataset
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if isinstance(outputs, dict):
            return outputs["last_hidden_state"][:, 0, :]  # [CLS] token
        else:
            return outputs.last_hidden_state[:, 0, :]  # [CLS] token
    
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
    # Initialize model
    model = SimCSEBERT()
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
    evaluator = STSEvaluator(model).to(device)
    
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
            emb1 = model(input_ids, attention_mask)  # First pass
            emb2 = model(input_ids, attention_mask)  # Second pass (different dropout)
            
            loss = model.simcse_loss(emb1, emb2, temperature=args.temperature)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Evaluate on STS after each epoch
        sts_corr = evaluate_sts_spearman(evaluator, sts_dev_dataloader, device)
        print(f"Epoch {epoch+1} - STS Spearman Correlation: {sts_corr:.4f}")
        
        if sts_corr > best_sts_corr:
            best_sts_corr = sts_corr
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, f"models/simcse_nli/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            print(f"New best model saved with Spearman: {sts_corr:.4f}")
    
    # Final evaluation
    dev_corr = evaluate_sts_spearman(evaluator, sts_dev_dataloader, device)
    print(f"Final Dev STS Spearman Correlation: {dev_corr:.4f}")
    
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
    parser.add_argument("--local_files_only", action="store_true", help="Use only local model files")
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("models/simcse_nli", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    train_simcse(args)