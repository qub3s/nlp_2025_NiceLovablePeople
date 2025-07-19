import argparse
import random

import numpy as np
import pandas as pd
import torch
import sklearn
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel
from sklearn.metrics import matthews_corrcoef
from optimizer import AdamW


TQDM_DISABLE = False

batch_size = 1 


class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=False)
        self.classifier = nn.Linear(self.bart.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        logits = self.classifier(cls_output)

        # Return the probabilities
        probabilities = self.sigmoid(logits)
        return probabilities


def transform_data(dataset, max_length=512):
    """
    dataset: pd.DataFrame

    Turn the data to the format you want to use.

    1. Extract the sentences from the dataset. We recommend using the already split
    sentences in the dataset.
    2. Use the AutoTokenizer from_pretrained to tokenize the sentences and obtain the
    input_ids and attention_mask.
    3. Currently, the labels are in the form of [6, 6, 6, 25, 25, 29]. This means that
    the sentence pair contains type 6, 25, and 29. Turn this into a binary form, where the
    label becomes [0, 0, 0, 0, 0, 1, ..., 1, 0, 0, 1, 0, 0].
    IMPORTANT: You will find that the dataset contains types up to 31, but some are not
    assigned. You need to drop 12, 19, 20, 23 and 27 when creating the binary labels.
    This way you should end up with a binary label of size 26.     
    Be careful that the test-student.csv does not
    have the paraphrase_types column. You should return a DataLoader without the labels.
    4. Use the input_ids, attention_mask, and binary labels to create a TensorDataset.
    Return a DataLoader with the TensorDataset. You can choose a batch size of your
    choice.
    """
    #raise NotImplementedError

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)


    # Tokenize the sentences
    s1 = dataset["sentence1_tokenized"].apply(eval)
    s2 = dataset["sentence2_tokenized"].apply(eval)

    # Tokenize the sentences *
    token = tokenizer(s1, s2, is_split_into_words=True, return_tensors="pt", padding=True)

    s1_atm = s1_token["attention_mask"]
    s1_inp_id = s1_token["input_ids"]


    if("paraphrase_type_ids" in dataset.keys()):
        # Read the solution labels
        para_ids = list(dataset["paraphrase_type_ids"].apply(eval)) 

        # Remove columns 12, 19, 20, 23, 27
        mask = torch.full( (32,), True)
        mask[[0, 12, 19, 20, 23, 27]] = False

        one_hot = []

        for x in para_ids:
            oh = nn.functional.one_hot(torch.tensor(x), 32)[:, mask]
            oh = torch.any(oh, dim=0)
            one_hot.append(oh)
        
        one_hot = torch.stack(one_hot)
        
        # create the Dataset and the Dataloader
        ds = TensorDataset(token["attention_mask"], token["input_ids"], one_hot)
        dl = DataLoader(ds, batch_size = batch_size, shuffle=True)
    else:
        # create the Dataset and the Dataloader
        ds = TensorDataset(token["attention_mask"], token["input_ids"])
        dl = DataLoader(ds, batch_size = batch_size, shuffle=True)

    return dl 


def train_model(model, train_data, dev_data, device, epochs=5, lr=0.01):
    """
    Train the model. You can use any training loop you want. We recommend starting with
    AdamW as your optimizer. You can take a look at the SST training loop for reference.
    Think about your loss function and the number of epochs you want to train for.
    You can also use the evaluate_model function to evaluate the
    model on the dev set. Print the training loss, training accuracy, and dev accuracy at
    the end of each epoch.

    Return the trained model.
    """
    ### TODO
    #raise NotImplementedError

    # create optimizer, loss_fn 
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    model.to(device)

    for epoch in range(epochs):
        model.train()

        # Reset vars
        train_loss = 0
        dev_loss = 0
        num_batches_train = 0
        num_batches_dev = 0

        for batch in tqdm(
            train_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            X, X_mask, Y = batch

            X = X.to(device)
            X_mask = X_mask.to(device)
            Y = Y.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # run predictions
            pred = model(X, X_mask)
            
            # calc loss
            loss = loss_fn(pred, Y.to(torch.float32))

            # backpass
            loss.backward()

            # adjust gradients
            optimizer.step()

            # log the loss 
            train_loss += loss.item()
            num_batches_train += 1

        model.eval()

        for batch in tqdm(
            dev_data, desc=f"train-{epoch+1:02}", disable=TQDM_DISABLE
        ):
            X, X_mask, Y = batch
            X_mask = X_mask.to(device)

            X = X.to(device)
            Y = Y.to(device)

            # do not collect gradients during validation
            with torch.no_grad():
                # make prediction
                pred = model(X, X_mask)

                # calc loss
                loss = loss_fn(pred, Y.to(torch.float32))

            # log the loss
            dev_loss += loss.item()
            num_batches_dev += 1

        print("Train Loss: ",train_loss/batch_size/num_batches_train)
        print("Validation Loss: ",dev_loss/batch_size/num_batches_dev)

    return model

def test_model(model, test_data, test_ids, device):
    """
    Test the model. Predict the paraphrase types for the given sentences and return the results in form of
    a Pandas dataframe with the columns 'id' and 'Predicted_Paraphrase_Types'.
    The 'Predicted_Paraphrase_Types' column should contain the binary array of your model predictions.
    Return this dataframe.
    """
    ### TODO
    #raise NotImplementedError

    model.to(device)

    # set the model to eval (not store gradients)
    model.eval()
    
    # load the data
    df = pd.DataFrame(columns=['id', 'Predicted_Paraphrase_Types'])

    
    for data, ids in zip(test_data, test_ids):

        X, X_mask = data 
        X = X.to(device)
        X_mask = X_mask.to(device)

        # do not collect gradients
        with torch.no_grad():
            # make prediction
            pred = model(X, X_mask)

        # insert into dataframe
        i = len(df)
        df.loc[i, 'id'] = ids
        df.loc[i, 'Predicted_Paraphrase_Types'] = pred 

    return df

def evaluate_model(model, test_data, device):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the 26 paraphrase types the model has predicted correctly for each data point..
    """
    all_pred = []
    all_labels = []
    model.eval()

    with torch.no_grad():
        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > 0.5).int()

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    all_predictions = torch.cat(all_pred, dim=0)
    all_true_labels = torch.cat(all_labels, dim=0)

    true_labels_np = all_true_labels.cpu().numpy()
    predicted_labels_np = all_predictions.cpu().numpy()

    # Compute the accuracy for each label
    accuracies = []
    matthews_coefficients = []
    for label_idx in range(true_labels_np.shape[1]):
        correct_predictions = np.sum(true_labels_np[:, label_idx] == predicted_labels_np[:, label_idx])
        total_predictions = true_labels_np.shape[0]
        label_accuracy = correct_predictions / total_predictions
        accuracies.append(label_accuracy)

        # compute Matthwes Correlation Coefficient for each paraphrase type
        matth_coef = matthews_corrcoef(true_labels_np[:, label_idx], predicted_labels_np[:, label_idx])
        matthews_coefficients.append(matth_coef)

    # Calculate the average accuracy over all labels
    accuracy = np.mean(accuracies)
    matthews_coefficient = np.mean(matthews_coefficients)
    model.train()
    return accuracy, matthews_coefficient


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


def finetune_paraphrase_detection(args):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")
    

    # TODO You might do a split of the train data into train/validation set here
    # (or in the csv files directly)
    train_ds, val_ds = sklearn.model_selection.train_test_split(train_dataset, test_size=0.2)
    
    train_data = transform_data(train_ds)
    dev_data = transform_data(val_ds)
    test_data = transform_data(test_dataset)

    print(f"Loaded {len(train_dataset)} training samples.")

    #model = train_model(model, train_data, dev_data, device)

    #print("Training finished.")

    
    #accuracy, matthews_corr = evaluate_model(model, dev_data, device)
    #print(f"The accuracy of the model is: {accuracy:.3f}")
    #print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    finetune_paraphrase_detection(args)
