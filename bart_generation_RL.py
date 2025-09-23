import numpy as np
from torch import nn
import torch.nn.functional as F
import pandas as pd
import torch
import sklearn
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartForConditionalGeneration

import matplotlib.pyplot as plt
from bart_generation_earlystopping import PGEarlyStopping

from optimizer import AdamW

from tokenizer import BertTokenizer

TQDM_DISABLE = False

class GeneratorEvaluator():

    def __init__(self, generator, generator_tokenizer, evaluator, evaluator_tokenizer, device):
        #super().__init__() # TODO
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.evaluator = evaluator
        self.evaluator_tokenizer = evaluator_tokenizer
        self.device = device

    def get_similarity_score(gen_text, target_text):
        """
        Calculate semantic similarity score for prediction of generator. 
        Transform it into range of [0,1].
        Returns transformed score.
        """

        # This is what happens per batch, so a batch could be done without a for loop!!!
        # Data_loader needs to be prepared though -> check

        # Prepare data for bert model (evaluator)
        BertTokenizer.from_pretrained("bert-base-uncased") # FOr main
        token = tokenizer(batch, return_tensors="pt", padding=True) 
        b_input_ids = token["input_ids"]
        b_attention_mask = token["attention_mask"]

        # Get similarity score [1,5]
        b_ids1 = b_ids1.to(self.device)
        b_mask1 = b_mask1.to(self.device)
        b_ids2 = b_ids2.to(self.device)
        b_mask2 = b_mask2.to(self.device)

        logits = self.evaluator.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
        y_hat = logits.flatten().cpu().numpy()

        # Transform similarity score linearly into [0,1]
        similarity_score = y_hat / 5.0

        return similarity_score


    def train(epochs=10, lr=1e-05, dataloader_train):
        
        optimizer = AdamW(self.generator.parameters(), lr=lr, weight_decay=0.01) 
        self.generator.to(self.device)

        # Training loop
        for epoch in range(num_epochs):

            # Track rewards for monitoring
            epoch_rewards = [] 
            epoch_losses = []
            
            # Training
            self.generator.train()  

            for batch in tqdm(
                dataloader_train, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                # Prepare data
                b_input_ids = batch['input_ids'].to(self.device)
                b_attention_mask = batch['attention_mask'].to(self.device)
                b_labels = batch['labels'].to(self.device) 

                # Reset gradients
                optimizer.zero_grad()
                
                # Generate rollouts (sampling)
                # No gradients during generation
                with torch.no_grad():  
                    generated_outputs = self.generator.generate(
                        input_ids=b_input_ids,
                        attention_mask=b_attention_mask,
                        max_length=50,
                        do_sample=True,           # CRITICAL: Must sample for RL
                        num_return_sequences=1,   # Generate one sequence per input
                        temperature=0.8,          # Controls randomness (0.7-1.0 is good)
                        top_k=50,                 # Top-k sampling
                        pad_token_id=self.generator_tokenizer.pad_token_id,
                        eos_token_id=self.generator_tokenizer.eos_token_id, 
                        return_dict_in_generate=True,
                        output_scores=True,       # Need this for token probabilities
                        output_hidden_states=True # Need this for penalty if used
                    )
                
                # Get the generated token sequences
                generated_sequences = generated_outputs.sequences  # [batch_size, seq_len]
                
                # Step 2.3: Get token probabilities for REINFORCE
                # generated_outputs.scores is a tuple of [batch_size, vocab_size] for each generation step
                logits_per_step = generated_outputs.scores
                
                # Calculate log probability for each generated token
                log_probs = []
                for step, step_logits in enumerate(logits_per_step):
                    # Get the tokens actually generated at this step
                    tokens_at_this_step = generated_sequences[:, step + 1]  # +1 to skip start token
                    
                    # Calculate log probabilities for all tokens at this step
                    log_prob = F.log_softmax(step_logits, dim=-1)  # [batch_size, vocab_size]
                    
                    # Get the log prob of the specific tokens that were chosen
                    log_prob_chosen = log_prob.gather(dim=-1, index=tokens_at_this_step.unsqueeze(-1)).squeeze(-1)
                    log_probs.append(log_prob_chosen)
                
                # Stack across sequence length: [batch_size, seq_length]
                log_probs = torch.stack(log_probs, dim=1)
                
                # Step 2.4: Create mask for generated tokens (ignore padding)
                gen_mask = (generated_sequences != tokenizer.pad_token_id).int()[:, 1:]  # Remove first token
                # Ensure mask matches log_probs shape
                if gen_mask.shape[1] > log_probs.shape[1]:
                    gen_mask = gen_mask[:, :log_probs.shape[1]]
                
                # Calculate total log probability for each sequence
                sequence_log_prob = (log_probs * gen_mask).sum(dim=-1)  # [batch_size]
                
                # Calculate rewards using evaluator model
                rewards = []
                generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
                input_texts = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True)
                target_texts = tokenizer.batch_decode(b_labels, skip_special_tokens=True)
                
                for gen_text, target_text in zip(generated_texts, target_texts): #TODO: do it batch_wise
                    # Get similarity score from your evaluator model
                    similarity_score = get_similarity_score(gen_text, target_text)
                    
                    # Simple reward: just use the similarity score
                    reward = similarity_score
                    
                    # Optional: Add anti-copying penalty to reward
                    # reward = similarity_score - (copy_penalty_weight * copying_score)
                    
                    rewards.append(reward)
                
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                epoch_rewards.append(rewards.mean().item())
                
                # Calculate RL loss
                # Normalize rewards (reduce variance)
                reward_baseline = rewards.mean()
                adjusted_rewards = rewards - reward_baseline
                
                # RL loss: - (log_prob * advantage)
                reinforce_loss = - (sequence_log_prob * adjusted_rewards).mean()
                epoch_losses = reinforce_loss.item()

                # Backward pass and optimize
                reinforce_loss.backward()
                
                # Gradient clipping for RL stability
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                
                optimizer.step()
                
            # Logging
            tqdm.write(f"Epoch {epoch+1}\n Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")
            tqdm.write(f"Average Reward: {sum(epoch_rewards) / len(epoch_rewards):.4f}")
            
