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
import itertools
import json
from datetime import datetime

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
    evaluator = STSEvaluator(model)
    
    best_sts_corr = -1
    best_model_state = None
    patience_counter = 0
    
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
        
        # Early stopping check
        if sts_corr > best_sts_corr + 0.001:  # Minimum improvement
            best_sts_corr = sts_corr
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            torch.save(best_model_state, f"models/simcse_nli/best_model_epoch{epoch+1}_corr{sts_corr:.4f}.pt")
            print(f"New best model saved and Spearman: {sts_corr:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
        # Early stopping
        if patience_counter >= 2:  # Stop if no improvement for 2 epochs
            print("Early stopping triggered")
            break
    
    # Load best model for final evaluation
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    dev_corr = evaluate_sts_spearman(evaluator, sts_dev_dataloader, device)
    print(f"Final STS Spearman Correlation: {dev_corr:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), "models/simcse_nli/final_model.pt")
    print("Final model saved.")
    
    return model, dev_corr

def hyperparameter_search():
    """Grid search for optimal SimCSE parameters"""
    
    # Define parameter grid
    param_grid = {
        'batch_size': [32, 64, 128],
        'lr': [1e-5, 2e-5, 5e-5],
        'temperature': [0.05, 0.1, 0.2],
        'epochs': [2, 3, 4],
        'subset_size': [20000]  # Fixed subset size for search
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, combination)) 
                         for combination in itertools.product(*values)]
    
    results = []
    best_corr = -1
    best_params = None
    
    print(f"Starting hyperparameter search with {len(param_combinations)} combinations")
    
    for i, params in enumerate(param_combinations):
        print(f"\n{'='*50}")
        print(f"Testing combination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        print(f"{'='*50}")
        
        # Set args with current parameters
        args = argparse.Namespace()
        args.batch_size = params['batch_size']
        args.lr = params['lr']
        args.epochs = params['epochs']
        args.temperature = params['temperature']
        args.small_subset = True
        args.subset_size = params['subset_size']
        
        try:
            # Train with current parameters
            model, sts_corr = train_simcse(args)
            
            result = {
                'params': params,
                'sts_correlation': sts_corr,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Track best parameters
            if sts_corr > best_corr:
                best_corr = sts_corr
                best_params = params
                print(f"NEW BEST: STS Correlation = {sts_corr:.4f}")
            
            # Save results incrementally
            with open('models/simcse_nli/hparam_search_results.json', 'w') as f:
                json.dump({
                    'all_results': results,
                    'best_params': best_params,
                    'best_correlation': best_corr
                }, f, indent=2)
                
        except Exception as e:
            print(f"Error with parameters {params}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE!")
    print(f"Best STS Correlation: {best_corr:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"{'='*60}")
    
    return best_params, best_corr

def analyze_results():
    """Analyze hyperparameter search results"""
    try:
        with open('models/simcse_nli/hparam_search_results.json', 'r') as f:
            data = json.load(f)
        
        results = data['all_results']
        
        print("Hyperparameter Search Analysis")
        print("=" * 60)
        
        # Sort by performance
        sorted_results = sorted(results, key=lambda x: x['sts_correlation'], reverse=True)
        
        print("Top 5 Performers:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i+1}. Correlation: {result['sts_correlation']:.4f}")
            print(f"   Params: {result['params']}")
            print()
        
        # Analyze parameter importance
        param_analysis = {}
        for param in ['batch_size', 'lr', 'temperature', 'epochs']:
            values = {}
            for result in results:
                value = result['params'][param]
                if value not in values:
                    values[value] = []
                values[value].append(result['sts_correlation'])
            
            # Calculate average performance for each parameter value
            avg_performance = {v: sum(values[v])/len(values[v]) for v in values}
            param_analysis[param] = avg_performance
        
        print("Parameter Performance Analysis:")
        for param, performances in param_analysis.items():
            print(f"\n{param}:")
            for value, avg_corr in sorted(performances.items(), key=lambda x: x[1], reverse=True):
                print(f"  {value}: {avg_corr:.4f}")
                
    except FileNotFoundError:
        print("No results found. Run hyperparameter search first.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--small_subset", action="store_true", help="Use small subset for testing")
    parser.add_argument("--subset_size", type=int, default=20_000, help="Size of subset for hyperparameter tuning")
    parser.add_argument("--hparam_search", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--train_full", action="store_true", help="Train on full dataset with best params")
    parser.add_argument("--analyze", action="store_true", help="Analyze hyperparameter search results")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("models/simcse_nli", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    if args.hparam_search:
        print("Starting hyperparameter search...")
        best_params, best_corr = hyperparameter_search()
        
        # Save best parameters for full training
        with open('models/simcse_nli/best_params.json', 'w') as f:
            json.dump({
                'best_params': best_params,
                'best_correlation': best_corr,
                'search_date': datetime.now().isoformat()
            }, f, indent=2)
            
    elif args.train_full:
        # Load best parameters from previous search
        try:
            with open('models/simcse_nli/best_params.json', 'r') as f:
                best_data = json.load(f)
                best_params = best_data['best_params']
                
            print(f"Training with best parameters: {best_params}")
            
            # Set args for full training
            args.batch_size = best_params['batch_size']
            args.lr = best_params['lr']
            args.epochs = best_params['epochs']
            args.temperature = best_params['temperature']
            args.small_subset = False  # Use full dataset
            args.subset_size = None
            
            # Train on full dataset
            model, final_corr = train_simcse(args)
            
            print(f"Final training complete! STS Correlation: {final_corr:.4f}")
            
        except FileNotFoundError:
            print("No best parameters found. Run --hparam_search first.")
            
    elif args.analyze:
        analyze_results()
        
    else:
        # Default training (for testing)
        train_simcse(args)