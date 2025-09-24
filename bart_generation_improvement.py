import argparse
import random

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
from multitask_classifier import MultitaskBERT
from bart_generation_RL import GeneratorEvaluatorRL


TQDM_DISABLE = False
BATCH_SIZE = 32 


def transform_data(dataset, max_length=256, shuffle=True):
    """
    Turn the data to the format you want to use.
    Use AutoTokenizer to obtain encoding (input_ids and attention_mask).
    Tokenize the sentence pair in the following format:
    sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids.
    Return Data Loader.
    """
    ### TODO 
    #raise NotImplementedError
    
    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    # Get data for sentence pairs out of the dataset
    # Format it according to: sentence_1 + SEP + sentence_1 segment location + SEP + paraphrase_type_ids
    SEP = tokenizer.sep_token
    formatted_input = list(dataset.apply(lambda row: ' '.join([row['sentence1'], SEP, row['sentence1_segment_location'], SEP, row['paraphrase_type_ids']]), axis=1))
    
    # Get input_ids and attention_mask
    token = tokenizer(formatted_input, return_tensors="pt", padding=True) 
    input_ids = token["input_ids"]
    attention_mask = token["attention_mask"]

    # Get DataLoader
    print("Batch Size: ", BATCH_SIZE)

    # If not test set
    if ('sentence2' in dataset.keys()):
        formatted_target = list(dataset["sentence2"])
        target_token = tokenizer(formatted_target, return_tensors="pt", padding=True)
        dataset = TensorDataset(input_ids, attention_mask, target_token["input_ids"])
    else:
        dataset = TensorDataset(input_ids, attention_mask)

    # Combine into a TensorDataset
    dataloader = DataLoader(
            dataset,
            batch_size = BATCH_SIZE,
            shuffle = shuffle
        )
    return dataloader

### Train without Evaluator ###

def mean_pool(hidden_states, mask):
    # hidden_states: [batch, seq_len, hidden_dim]
    # mask: [batch, seq_len]
    input_mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_l(step, l_start=0.7, l_end=0.1):
    # exponential weight decay for l
    return l_end + (l_start - l_end) * (0.95 ** step)

def get_l_warmed_up(step, total_batches_estimate, warmup_frac=0.15, l_max=0.7, l_min=0.1):
    """Step is the current batch number."""
    warmup_batches = int(total_batches_estimate * warmup_frac)
    
    if step < warmup_batches:
        return l_min + (l_max - l_min) * (step / warmup_batches)
    else:
        decay_batches = step - warmup_batches
        total_decay_batches = total_batches_estimate - warmup_batches
        return l_max - (l_max - l_min) * min(decay_batches / total_decay_batches, 1.0)    

def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """

    dataloader_train = transform_data(train_data)
    dataloader_dev = transform_data(dev_data)

    lr = 1e-5
    epochs = 100 #TODO 
    # halfed total optimizer steps as early stopping on average roughly stops at half the time
    total_steps = epochs * len(train_data) * 0.7 
    frac = 0.15
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01) 
    cos_sim = nn.CosineEmbeddingLoss()
    model.to(device)

    filepath = f"models/finetune-paraphrase_generation-{lr}-{BATCH_SIZE}-FINAL"
    early_stop = PGEarlyStopping(filepath, patience=20, verbose=True, delta=0)

    dev_losses = []
    train_losses = []
    dev_bleu = []
    train_losses_penalised = []

    # Training loop
    for epoch in range(epochs):

        # Reset variables
        train_loss = 0
        train_loss_penalised = 0
        dev_loss = 0
        train_num_batches = 0
        dev_num_batches = 0

        # Training
        model.train()
        for batch in tqdm(
                dataloader_train, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):

            # Prepare data
            b_input_ids, b_attention_mask, b_labels = batch

            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_labels = b_labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Generate outputs
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels,
                output_hidden_states=True
            )

            # Get encoder embeddings (input representation)
            encoder_hidden = outputs.encoder_last_hidden_state  # [batch, seq_len, hidden_dim]
            input_embeds = mean_pool(encoder_hidden, b_attention_mask) # [batch, hidden_dim]

            # Get decoder embeddings (output representation)
            decoder_hidden = outputs.decoder_hidden_states[-1]  # last layer [batch, seq_len, hidden_dim]
            # Need a mask for generated labels (not attention mask)
            label_mask = (b_labels != tokenizer.pad_token_id).int()
            output_embeds = mean_pool(decoder_hidden, label_mask) # [batch, hidden_dim]
            
            # Calculate penalty
            target = torch.full((input_embeds.size(0),), -1.0, device=device)
            penalty = cos_sim(output_embeds, input_embeds, target)

            # Calculate penalised loss and optimise model
            #l = get_l(train_num_batches)
            l = get_l_warmed_up(step=train_num_batches, total_batches_estimate=total_steps, warmup_frac=frac)
            loss = (1-l) * outputs.loss + l * penalty
            loss.backward()
            optimizer.step()
            
            # Logging
            train_loss += outputs.loss.detach().float().cpu().item() #todo
            train_loss_penalised += loss.detach().float().cpu().item()
            train_num_batches += 1

            #break #TODO
        
        # Validation
        model.eval()
        for batch in tqdm(
                dataloader_dev, desc=f"dev-{epoch+1:02}", disable=TQDM_DISABLE
            ):
            # Prepare data
            b_input_ids, b_attention_mask, b_labels = batch

            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_labels = b_labels.to(device)

            # No gradients during validation
            with torch.no_grad():
                # Generate outputs
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels,
                )
                
                # Calculate loss
                loss = outputs.loss 
            
            # Logging
            dev_loss += outputs.loss.detach().float().cpu().item()
            dev_num_batches += 1
            #break #TODO
        
        bleu_score = evaluate_model(model, dev_data, device, tokenizer)

        # Log losses
        epoch_train_loss = train_loss / train_num_batches
        epoch_dev_loss = dev_loss / dev_num_batches
        dev_losses.append(epoch_dev_loss)
        train_losses.append(epoch_train_loss)
        epoch_train_loss_penalised = train_loss_penalised / train_num_batches
        dev_bleu.append(bleu_score)
        train_losses_penalised.append(epoch_train_loss_penalised)
        tqdm.write(f"Epoch {epoch+1}\t Train Loss: {epoch_train_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Validation Loss: {epoch_dev_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Train Loss Penalised: {epoch_train_loss_penalised:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Validation Penalised Bleu Score: {bleu_score:.4f}")
        #tqdm.write(f"Epoch {epoch+1}\t Validation Loss Penalised: {epoch_dev_loss_penalised:.4f}")

        # check for early stopping
        if early_stop(bleu_score, model, epoch):
            break
    
    print("LR: ", lr)
    print(f"Penalty loss used with warmup l, frac {frac}")

    # Plot loss over time
    epochs_plot = range(1, epochs + 1)
    plt.plot(epochs_plot, train_losses, 'o', label='Training loss')
    plt.plot(epochs_plot, dev_losses, 'o', label='Validation loss')
    plt.plot(epochs_plot, train_losses_penalised, 'o', label='Training loss penalised')
    #plt.plot(epochs_plot, dev_losses_penalised, 'o', label='Validation loss penalised')
    plt.title('Training and validation loss with and without penalty')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/losses_plot_{lr}_{BATCH_SIZE}_{frac}_FINAL.png", bbox_inches='tight')
    plt.close()

    # Plot Bleu Score over time
    plt.plot(epochs_plot, dev_bleu, 'o', label='Validation penalised BLEU score')
    plt.title('Validation penalised BLEU score')
    plt.xlabel('Epochs')
    plt.ylabel('Penalised BLEU score')
    plt.savefig(f"plots/bleu_plot_{lr}_{BATCH_SIZE}_{frac}_FINAL.png", bbox_inches='tight')

    return model



def test_model(test_data, test_ids, device, model, tokenizer):
    """
    Test the model. Generate paraphrases for the given sentences (sentence1) and return the results
    in form of a Pandas dataframe with the columns 'id' and 'Generated_sentence2'.
    The data format in the columns should be the same as in the train dataset.
    Return this dataframe.
    """
    ### TODO
    #raise NotImplementedError
    
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in test_data:
            # Prepare data
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate output
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            
            # Decode predictions
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]

            predictions.extend(pred_text)

    # Create dataframe with ids and predicted paraphrases
    data = {"id": test_ids, "Generated_sentence2": predictions} 
    df = pd.DataFrame(data=data)

    return df

def evaluate_model(model, test_data, device, tokenizer):
    """
    You can use your train/validation set to evaluate models performance with the BLEU score.
    test_data is a Pandas Dataframe, the column "sentence1" contains all input sentence and 
    the column "sentence2" contains all target sentences
    """
    model.eval()
    bleu = BLEU()
    predictions = []

    dataloader = transform_data(test_data, shuffle=False)
    with torch.no_grad():
        for batch in dataloader: 
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            # Generate paraphrases
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=50,
                num_beams=5,
                early_stopping=True,
            )
            
            pred_text = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for g in outputs
            ]
            
            predictions.extend(pred_text)

    inputs = test_data["sentence1"].tolist()
    references = test_data["sentence2"].tolist()

    model.train()
    # Calculate BLEU score
    bleu_score_reference = bleu.corpus_score(references, [predictions]).score
    # Penalize BLEU score if its to close to the input
    bleu_score_inputs = 100 - bleu.corpus_score(inputs, [predictions]).score

    print(f"BLEU Score: {bleu_score_reference}", f"Negative BLEU Score with input: {bleu_score_inputs}")
    

    # Penalize BLEU and rescale it to 0-100
    # If you perfectly predict all the targets, you should get an penalized BLEU score of around 52
    penalized_bleu = bleu_score_reference * bleu_score_inputs / 52
    print(f"Penalized BLEU Score: {penalized_bleu}")

    return penalized_bleu


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    return args


def finetune_paraphrase_generation(args):
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", local_files_only=True)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", local_files_only=True)

    # Prepare evaluator for RL
    #saved_evaluator = torch.load("models/final_sbert_finetune-5-2e-05-sts.pt", map_location=device)
    #saved_evaluator["model_config"].simcse_model_path = "models/final_sbert_finetune-5-2e-05-sts.pt"
    #config = saved_evaluator["model_config"]
    #evaluator = MultitaskBERT(config)
    #evaluator.load_state_dict(saved_evaluator["model"])

    #evaluator_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    # Prepare data
    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")

    # You might do a split of the train data into train/validation set here
    train_dataset, dev_dataset = sklearn.model_selection.train_test_split(train_dataset, test_size=0.2)

    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} ETPC training samples.")

    dataloader_train_RL = transform_data(train_dataset, shuffle=True)

    # Train RL
    #trainer_RL = GeneratorEvaluatorRL(model, tokenizer, evaluator, evaluator_tokenizer, device)
    #model = trainer_RL.train(dataloader_train=dataloader_train_RL, epochs=10, lr=1e-5) # TODO add DEV

    #print("Training with Evaluator finished.")

    model = train_model(model, train_dataset, dev_dataset, device, tokenizer)

    print("Training finished.")

    bleu_score = evaluate_model(model, dev_dataset, device, tokenizer)
    print(f"The penalized BLEU-score of the model is: {bleu_score:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(test_data, test_ids, device, model, tokenizer)
    test_results.to_csv(
        "predictions/bart/etpc-paraphrase-generation-test-output.csv", index=False
    )


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_generation(args)
