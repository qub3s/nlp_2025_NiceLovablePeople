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

from optimizer import AdamW


TQDM_DISABLE = False


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
    # TODO 
    input_for_loss = list(dataset["sentence1"])
    input_for_loss = tokenizer(input_for_loss, return_tensors="pt", max_length=max_length, padding="max_length")["input_ids"]
    # Get input_ids and attention_mask
    token = tokenizer(formatted_input, return_tensors="pt", max_length=max_length, padding="max_length") 
    input_ids = token["input_ids"]
    attention_mask = token["attention_mask"]

    # Get DataLoader
    batch_size = 1#8 #TODO
    print("Batch Size: ", batch_size)

    # If not test set
    if ('sentence2' in dataset.keys()):
        formatted_target = list(dataset["sentence2"])
        target_token = tokenizer(formatted_target, return_tensors="pt", max_length=max_length, padding="max_length")
        dataset = TensorDataset(input_ids, attention_mask, target_token["input_ids"], input_for_loss)
    else:
        dataset = TensorDataset(input_ids, attention_mask)

    # Combine into a TensorDataset
    dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )
    return dataloader

def train_model(model, train_data, dev_data, device, tokenizer):
    """
    Train the model. Return and save the model.
    """
    ### TODO
    #raise NotImplementedError
    max_length=256
    l = 1 #TODO
    lr = 1e-5
    epochs = 2#5  #TODO 
    print("Epochs: ", epochs)
    optimizer = AdamW(model.parameters(), lr=lr) #lr = 2e-5, eps = 1e-8 is default
    model.to(device)

    dev_losses = []
    train_losses = []
    dev_losses_penalised = []
    train_losses_penalised = []

    # Training loop
    for epoch in range(epochs):

        # Reset variables
        train_loss = 0
        train_loss_penelised = 0
        dev_loss = 0
        dev_loss_penalised = 0
        train_num_batches = 0
        dev_num_batches = 0

        # Training
        model.train()
        for batch in tqdm(
                train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
            ):

            # Prepare data
            b_input_ids, b_attention_mask, b_labels, input_for_loss = batch

            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_labels = b_labels.to(device)
            input_for_loss = input_for_loss.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Generate outputs
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask,
                labels=b_labels
            )

            # Prepare predictions for penalty calculation
            predicted_ids = outputs.logits.argmax(-1)
            padding = input_for_loss.shape #- predicted_ids.shape.item()
            #print("Padding:", (0, input_for_loss.shape[1]-predicted_ids.shape[1]))
            padded_predicted_ids = F.pad(predicted_ids, (0, max_length-predicted_ids.shape[1]), mode='constant', value=1)
            #print("Predicted_ids shape:", predicted_ids.shape)
            #print("Padded Predicted_ids:", padded_predicted_ids.shape)
            
            # Calculate penalty
            cos_sim = nn.CosineEmbeddingLoss()
            target = torch.ones(padding[0]) * -1
            target = target.to(device)
            input_similarity = cos_sim(padded_predicted_ids.to(torch.float32), input_for_loss.to(torch.float32), target)
            #print("Input_sim: ", input_similarity)

            # Calculate penalised loss and optimise model
            loss = outputs.loss + l * input_similarity
            #print("loss:", loss)
            loss.backward()
            optimizer.step()
            # Logging
            train_loss += outputs.loss.detach().float() #todo
            train_loss_penelised = loss.detach().float()
            train_num_batches += 1
            break #TODO
        
        # Validation
        model.eval()
        for batch in tqdm(
                dev_data, desc=f"dev-{epoch+1:02}", disable=TQDM_DISABLE
            ):
            # Prepare data
            b_input_ids, b_attention_mask, b_labels, input_for_loss = batch

            b_input_ids = b_input_ids.to(device)
            b_attention_mask = b_attention_mask.to(device)
            b_labels = b_labels.to(device)
            input_for_loss = input_for_loss.to(device)

            # No gradients during validation
            with torch.no_grad():
                # Generate outputs
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels,
                )
                
                # Prepare predictions for penalty calculation
                predicted_ids = outputs.logits.argmax(-1)
                padding = input_for_loss.shape #- predicted_ids.shape.item()
                print("Padding:", (input_for_loss.shape[1]-predicted_ids.shape[1]))
                padded_predicted_ids = F.pad(predicted_ids, (0, max_length-predicted_ids.shape[1]), mode='constant', value=1)
                
                # Calculate penalty
                cos_sim = nn.CosineEmbeddingLoss()
                target = torch.ones(padding[0]) * -1
                target = target.to(device)
                input_similarity = cos_sim(padded_predicted_ids.to(torch.float32), input_for_loss.to(torch.float32), target)
                print("validation Penalty: ", input_similarity)
                # Calculate penalised loss
                loss = outputs.loss + l * input_similarity
            
            # Logging
            dev_loss += outputs.loss.detach().float()
            dev_loss_penalised += loss.detach().float()
            dev_num_batches += 1
            break #TODO
        
        # Log losses
        epoch_train_loss = train_loss / train_num_batches
        epoch_dev_loss = dev_loss / dev_num_batches
        dev_losses.append(epoch_dev_loss)
        train_losses.append(epoch_train_loss)
        epoch_train_loss_penalised = train_loss_penelised / train_num_batches
        epoch_dev_loss_penalised = dev_loss_penalised / dev_num_batches
        dev_losses_penalised.append(epoch_dev_loss_penalised)
        train_losses_penalised.append(epoch_train_loss_penalised)
        tqdm.write(f"Epoch {epoch+1}\t Train Loss: {epoch_train_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Validation Loss: {epoch_dev_loss:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Train Loss Penalised: {epoch_train_loss_penalised:.4f}")
        tqdm.write(f"Epoch {epoch+1}\t Validation Loss Penalised: {epoch_dev_loss_penalised:.4f}")

    # Plot loss over time
    epochs_plot = range(1, epochs + 1)
    plt.plot(epochs_plot, train_losses, 'o', label='Training loss')
    plt.plot(epochs_plot, dev_losses, 'o', label='Validation loss')
    plt.plot(epochs_plot, train_losses_penalised, 'o', label='Training loss penalised')
    plt.plot(epochs_plot, dev_losses_penalised, 'o', label='Validation loss penalised')
    plt.title('Training and validation loss with and without penalty')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("plots/losses_plot.png", bbox_inches='tight')

    filepath = f"models/baseline-{epochs}-{lr}-paraphrase_detection.pt"
    torch.save(model, filepath)
    print(f"Saving the model to {filepath}.")

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
            input_ids, attention_mask, _, _ = batch # TODO ,_ remove
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

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-generation-test-student.csv")
    #dev_dataset = pd.read_csv("data/etpc-paraphrase-dev.csv")

    # You might do a split of the train data into train/validation set here
    # ...
    train_dataset, dev_dataset = sklearn.model_selection.train_test_split(train_dataset, test_size=0.2)

    train_data = transform_data(train_dataset)
    dev_data = transform_data(dev_dataset) 
    test_data = transform_data(test_dataset, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples.")

    model = train_model(model, train_data, dev_data, device, tokenizer)

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
