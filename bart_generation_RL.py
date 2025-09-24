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

class GeneratorEvaluatorRL():

    def __init__(self, generator, generator_tokenizer, evaluator, evaluator_tokenizer, device):
        super().__init__() # TODO
        self.generator = generator
        self.generator_tokenizer = generator_tokenizer
        self.evaluator = evaluator
        self.evaluator_tokenizer = evaluator_tokenizer 
        self.device = device


    # Based on preprocess function in datasets
    def preprocess_batch(self, strings):
        """Vectorized preprocessing using pandas."""
        series = pd.Series(strings)
        # Vectorized string operations
        processed = (series.str.lower()
                    .str.replace(r"([.,?\'])", r" \1", regex=True)  # Add space before punctuation
                    .str.split()
                 .str.join(" "))
        return processed.tolist()

    def data_for_evaluator(self, gen_texts, target_texts):

        # preprocessing use datasets.py function
        gen_texts = self.preprocess_batch(gen_texts)
        target_texts = self.preprocess_batch(target_texts)

        gen_token = self.evaluator_tokenizer(gen_texts, return_tensors="pt", padding=True, truncation=True)
        target_token = self.evaluator_tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True)

        gen_token_ids = torch.LongTensor(gen_token["input_ids"])
        gen_attention_mask = torch.LongTensor(gen_token["attention_mask"])
        target_token_ids = torch.LongTensor(target_token["input_ids"])
        target_attention_mask = torch.LongTensor(target_token["attention_mask"])

        return gen_token_ids, gen_attention_mask, target_token_ids, target_attention_mask


    def get_similarity_score(self, gen_texts, target_texts):
        """
        Calculate semantic similarity score for prediction of generator. 
        Transform it into range of [0,1].
        Returns transformed score.
        """

        # Prepare the data for the evaluator
        gen_token_ids, gen_attention_mask, target_token_ids, target_attention_mask = self.data_for_evaluator(gen_texts, target_texts)

        # Get similarity score [1,5]
        gen_token_ids = gen_token_ids.to(self.device)
        gen_attention_mask = gen_attention_mask.to(self.device)
        target_token_ids = target_token_ids.to(self.device)
        target_attention_mask = target_attention_mask.to(self.device)

        # No gradient updates for the evaluator
        with torch.no_grad():
            logits = self.evaluator.predict_similarity(gen_token_ids, gen_attention_mask, target_token_ids, target_attention_mask)
            y_hat = logits.flatten() # .cpu().numpy() #TODO necessary?

        # Transform similarity score linearly into [0,1]
        similarity_score = y_hat / 5.0
        print("SIMILARITY SCORE: ", similarity_score)
        return similarity_score.detach()

    def sample_sequences_one_forward(self, input_ids, attention_mask, max_length=50):
        """Sample using a single forward pass with causal masking."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create initial decoder input (just start tokens)
        decoder_input_ids = torch.tensor(
            [self.generator_tokenizer.bos_token_id] * batch_size, 
            device=device
        ).unsqueeze(1)
        
        all_logits = []
        all_tokens = [decoder_input_ids]
        
        for step in range(max_length - 1):
            # Forward pass
            outputs = self.generator(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            
            # Get the last token's logits
            next_token_logits = outputs.logits[:, -1, :]
            all_logits.append(next_token_logits)
            
            # Sample next token (with temperature and top-k)
            next_token_logits = next_token_logits / 0.8  # temperature
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(probs, 50, dim=-1)
            filtered_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            next_tokens = torch.multinomial(filtered_probs, num_samples=1)
            next_tokens = top_k_indices.gather(dim=-1, index=next_tokens)
            
            # Update decoder input for next step (only keep last token for efficiency)
            decoder_input_ids = next_tokens
            all_tokens.append(next_tokens)
        
        # Combine all generated tokens
        generated_sequences = torch.cat(all_tokens, dim=1)
        return generated_sequences, all_logits


    def train(self, dataloader_train, epochs=10, lr=1e-05):
        
        optimizer = AdamW(self.generator.parameters(), lr=lr, weight_decay=0.01) 
        self.generator.to(self.device)

        # Training loop
        for epoch in range(epochs):

            # Track rewards for monitoring
            epoch_rewards = [] 
            epoch_losses = []
            
            # Training
            self.generator.train()  

            for batch in tqdm(
                dataloader_train, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):
                # Prepare data
                b_input_ids, b_attention_mask, b_labels = batch

                b_input_ids = b_input_ids.to(self.device)
                b_attention_mask = b_attention_mask.to(self.device)
                b_labels = b_labels.to(self.device)

                # Reset gradients
                optimizer.zero_grad()
                
                # Generate rollouts (sampling)
                generated_sequences, logits_per_step  = self.sample_sequences_one_forward(b_input_ids, b_attention_mask)

                print(f"Gradients enabled: {all_logits[0].requires_grad}") 

                valid_tokens = (generated_sequences >= 0) & (generated_sequences < self.generator.config.vocab_size)
                if not valid_tokens.all():
                    print(f"ERROR: {(~valid_tokens).sum().item()} invalid tokens in generated sequences!")
                
                # Calculate log probability for each generated token
                log_probs = []
                for step, step_logits in enumerate(logits_per_step):
                    if step + 1 >= generated_sequences.size(1):
                        break  
                    # Get the tokens actually generated at this step
                    tokens_at_this_step = generated_sequences[:, step + 1]  # +1 to skip start token
                    
                    # Calculate log probabilities for all tokens at this step
                    #log_prob = F.log_softmax(step_logits, dim=-1)  # [batch_size, vocab_size]
                    probs = F.softmax(step_logits, dim=-1)            # [batch, vocab]
                    probs = probs.clamp(min=1e-12)                    # avoid exact zeros
                    log_prob = probs.log()
                    
                    # Get the log prob of the specific tokens that were chosen
                    log_prob_chosen = log_prob.gather(dim=-1, index=tokens_at_this_step.unsqueeze(-1)).squeeze(-1)
                    log_probs.append(log_prob_chosen)

                    if torch.isinf(log_prob_chosen).any():
                        print(f"WARNING: -inf detected at step {step}")
                        print(f"Tokens with -inf: {tokens_at_this_step[torch.isinf(log_prob_chosen)]}")
                        # Check what probabilities these tokens had
                        problematic_indices = torch.isinf(log_prob_chosen)
                        problematic_tokens = tokens_at_this_step[problematic_indices]
                        problematic_logits = step_logits[problematic_indices]
                        print(f"Problematic tokens: {problematic_tokens}")
                        print(f"Their logits: {problematic_logits[:, problematic_tokens]}")

                # Stack across sequence length: [batch_size, seq_length]
                log_probs = torch.stack(log_probs, dim=1)
                
                # Create mask for generated tokens (ignore padding)
                gen_mask = (generated_sequences != self.generator_tokenizer.pad_token_id).int()[:, 1:]  # Remove first token
                # Ensure mask matches log_probs shape
                if gen_mask.shape[1] > log_probs.shape[1]:
                    gen_mask = gen_mask[:, :log_probs.shape[1]]

                sequence_lengths = gen_mask.sum(dim=-1).float()  # Actual length of each sequence

                # Calculate average log prob per token instead of total
                sequence_log_prob = (log_probs * gen_mask).sum(dim=-1) / sequence_lengths.clamp(min=1) # [batch_size]

                
                # Calculate rewards using evaluator model
                rewards = []
                generated_texts = self.generator_tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
                #input_texts = tokenizer.batch_decode(b_input_ids, skip_special_tokens=True) #TODO not needed
                target_texts = self.generator_tokenizer.batch_decode(b_labels, skip_special_tokens=True)
                
                # Simple reward: just use the similarity score #TODO: more complicated reward system?
                rewards = self.get_similarity_score(generated_texts,  target_texts)

                # Optional: Add anti-copying penalty to reward
                # rewards = similarity_score - (copy_penalty_weight * copying_score)
               
                epoch_rewards.append(rewards.mean().item())
                
                # Calculate RL loss
                # Normalize rewards (reduce variance)
                reward_baseline = rewards.mean()
                adjusted_rewards = rewards - reward_baseline
                
                # RL loss: - (log_prob * advantage)
                print("Sequence log prob: ", sequence_log_prob)
                print("rewards: ", adjusted_rewards)
                reinforce_loss = - (sequence_log_prob * adjusted_rewards).mean()
                epoch_losses.append(reinforce_loss.item())

                # Backward pass and optimize
                reinforce_loss.backward()
                
                # Gradient clipping for RL stability
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                
                optimizer.step()

                break #TODO
                
            # Logging
            tqdm.write(f"Epoch {epoch+1}\n Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")
            tqdm.write(f"Average Reward: {sum(epoch_rewards) / len(epoch_rewards):.4f}")

        return self.generator
            
