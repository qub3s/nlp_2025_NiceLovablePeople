import argparse
import random
import math

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
import torch.nn.functional as F

TQDM_DISABLE = False

batch_size = 32
lr = 5e-05
epochs = 100 

class Focal_Loss(nn.Module):
    def __init__(self, alphas, gamma):
        super(Focal_Loss, self).__init__()
        self.alphas = torch.tensor(alphas)
        self.gamma = gamma
        
    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy(input, target, reduction='none')

        pt = torch.where(target == 1, input, 1 - input)

        focal_weight = self.alphas * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss

        return torch.sum(loss)

class Weight_based_sampler(torch.utils.data.Sampler):
    def __init__(self, data, distribution, seed=42):

        data = list(data["paraphrase_type_ids"].apply(eval)) 

        print(type(data))
        print(type(distribution))

        self.data = data
        self.seed = seed

        calc = lambda targets, distro: sum( x * y for x, y in zip(targets, distro)) / sum(targets)

        prob = []
        for x in data:
            prob.append(calc(x, distribution))

        s_prob = sum(prob)

        prob = [ x / s_prob for x in prob]

        self.probability = prob

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        iter_list = np.random.choice(np.arange(len(self)), size=len(self), replace=True, p=self.probability)

        return iter(iter_list)

class BartWithClassifier(nn.Module):
    def __init__(self, num_labels=26):
        super(BartWithClassifier, self).__init__()

        self.bart = BartModel.from_pretrained("facebook/bart-large", local_files_only=False)
        self.ln0 = nn.Linear(self.bart.config.hidden_size,  num_labels)
        self.ln1 = nn.Linear(self.bart.config.hidden_size,  128)
        self.ln2 = nn.Linear(128,  64)
        self.ln3 = nn.Linear(64, num_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids, attention_mask=None):
        # Use the BartModel to obtain the last hidden state
        outputs = self.bart(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]

        # Add an additional fully connected layer to obtain the logits
        
        logits = self.ln0(cls_output)

        #out_ln1 = self.dropout(self.relu(self.ln1(cls_output)))
        #out_ln2 = self.dropout(self.relu(self.ln2(out_ln1)))
        #logits = self.relu(self.ln3(out_ln2))

        # Return the probabilities
        probabilities = self.sigmoid(logits)

        return probabilities

def transform_data(dataset, max_length=512, shuffle=True, custom_sampler=None ):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)

    # Tokenize the sentences
    s1 = list(dataset["sentence1_tokenized"].apply(eval))
    s2 = list(dataset["sentence2_tokenized"].apply(eval))

    # Tokenize the sentences
    token = tokenizer(s1, s2, is_split_into_words=True, padding=True, return_tensors="pt")
    inp_ids = list(token["input_ids"])
    att_mask = list(token["attention_mask"])

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
        inp_ids = torch.stack(inp_ids)
        att_mask = torch.stack(att_mask)
        
        # create the Dataset and the Dataloader
        ds = TensorDataset(token["input_ids"], token["attention_mask"], one_hot)
        dl = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)

        if custom_sampler is None:
            dl = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)
        else:
            dl = DataLoader(ds, batch_size = batch_size, sampler=custom_sampler)

    else:
        # create the Dataset and the Dataloader
        ds = TensorDataset(token["input_ids"], token["attention_mask"])

        if custom_sampler is None:
            dl = DataLoader(ds, batch_size = batch_size, shuffle=shuffle)
        else:
            dl = DataLoader(ds, batch_size = batch_size, sampler=custom_sampler)

    return dl 

def train_model(model, train_data, dev_data, device, epochs=5, lr=lr, class_frequencies=[], gamma=0, name=""):
    
    # create optimizer, loss_fn 
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    #loss_fn = Focal_Loss(torch.tensor(class_frequencies).to(device), gamma)
    early_stopping = EarlyStopping("models/"+name+"_pd_model.pth", patience=5)
    model = model.to(device)

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

            Y = Y.to(torch.float32)
            Y = Y.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # run predictions
            pred = model(X, X_mask)
            
            # calc loss & clamp the values

            loss = loss_fn(pred, Y)

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
            Y = Y.to(torch.float32)
            Y = Y.to(device)

            # do not collect gradients during validation
            with torch.no_grad():
                # make prediction
                pred = model(X, X_mask)

                # calc loss
                eps = 1e-7
                pred = pred.clamp(min=eps, max=1-eps)
                Y = Y.clamp(min=eps, max=1-eps)

                loss = loss_fn(pred, Y)

            # log the loss
            dev_loss += loss.item()
            num_batches_dev += 1

        early_stopping(dev_loss, model);

        if early_stopping.early_stop:
            print("Early Stopp !!!")
            break

        print("Train Loss: ",train_loss/batch_size/num_batches_train)
        print("Validation Loss: ",dev_loss/batch_size/num_batches_dev)

    return model

def test_model(model, test_data, test_ids, device):
    model = model.to(device)

    # set the model to eval (not store gradients)
    model.eval()
    
    # load the data
    df = pd.DataFrame(columns=['id', 'Predicted_Paraphrase_Types'])
    
    c = 0

    for data in test_data:
        X, X_mask = data 
        X = X.to(device)
        X_mask = X_mask.to(device)

        # do not collect gradients
        with torch.no_grad():
            # make prediction
            pred = model(X, X_mask)

        # Threshhold the data
        pred = (pred > 0.5).int()

        # insert into dataframe
        for x in pred:
            i = len(df)
            df.loc[i, 'id'] = test_ids[c]
            df.loc[i, 'Predicted_Paraphrase_Types'] = x.tolist()
            c += 1

    return df

def evaluate_model(model, test_data, device, border):
    """
    This function measures the accuracy of our model's prediction on a given train/validation set
    We measure how many of the 26 paraphrase types the model has predicted correctly for each data point..
    """
    all_pred = []
    all_labels = []

    tp = []
    fp = []
    tn = []
    fn = []

    model.eval()

    with torch.no_grad():

        for batch in test_data:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predicted_labels = (outputs > border).int()

            predicted_labels = predicted_labels.to("cpu");

            tp.append(torch.sum(torch.logical_and(predicted_labels, labels)).item())
            fp.append(torch.sum(torch.logical_and(predicted_labels, torch.logical_not(labels))).item())

            tn.append(torch.sum(torch.logical_and(torch.logical_not(predicted_labels), torch.logical_not(labels))).item())
            fn.append(torch.sum(torch.logical_and(torch.logical_not(predicted_labels), labels)).item())

            all_pred.append(predicted_labels)
            all_labels.append(labels)

    total_tp = sum(tp)
    total_fp = sum(fp)
    total_tn = sum(tn)
    total_fn = sum(fn)

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * total_tp / (2 * total_tp + total_fp + total_fn +1e-8)

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

    return precision, recall, f1, accuracy, matthews_coefficient

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.model_checkpoint_path = checkpoint_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.model_checkpoint_path)
        self.val_loss_min = val_loss

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


def finetune_paraphrase_detection(args, iter_x, name):
    model = BartWithClassifier()
    device = torch.device("cuda") if args.use_gpu else torch.device("cpu")
    model.to(device)

    train_dataset = pd.read_csv("data/etpc-paraphrase-train.csv")
    test_dataset = pd.read_csv("data/etpc-paraphrase-detection-test-student.csv")

    # Get the class distribution
    train_dataset_label = []
    train_data_class_distribution = [0] * 32 


    for x in list(train_dataset["paraphrase_type_ids"].apply(eval)):
        train_dataset_label.append(set(x))

    for td in train_dataset_label:
        for x in range(32):
            if x in td:
                train_data_class_distribution[x] += 1

    train_data_class_distribution = [ train_data_class_distribution[x] for x in range(len(train_data_class_distribution)) if x not in [0, 12, 19, 20, 23, 27]]
    print(train_data_class_distribution)

    inverse_relative_class_frequencies = [ 1 / (x + 1e-8)**(1/4) for x in train_data_class_distribution]
    #inverse_relative_class_frequencies = [1] * 26 

    train_ds, val_ds = sklearn.model_selection.train_test_split(train_dataset, test_size=0.20)
    
    sampler = Weight_based_sampler(train_ds, inverse_relative_class_frequencies)
    #train_data = transform_data(train_ds, custom_sampler=sampler)

    train_data = transform_data(train_ds, shuffle=True)
    dev_data = transform_data(val_ds, shuffle = False)
    test_data = transform_data(test_dataset, shuffle = False)

    print(f"Loaded {len(train_dataset)} training samples.")

    print(lr)
    print(batch_size)
    print(epochs)
    print(name)

    train_model(model, train_data, dev_data, device, epochs=epochs, class_frequencies=inverse_relative_class_frequencies, gamma=3, name=name)
    model = model.to(torch.device("cpu"))
    model.load_state_dict(torch.load("models/"+name+"_pd_model.pth", map_location=torch.device("cpu")))
    model = model.to(device)

    print("Training finished.")

    precision, recall, f1, accuracy, matthews_corr = evaluate_model(model, dev_data, device, x)

    print(f"The precision of the model is: {precision:.3f}")
    print(f"The recall of the model is: {recall:.3f}")
    print(f"The f1 of the model is: {f1:.3f}")

    print(f"The accuracy of the model is: {accuracy:.3f}")
    print(f"Matthews Correlation Coefficient of the model is: {matthews_corr:.3f}")

    test_ids = test_dataset["id"]
    test_results = test_model(model, test_data, test_ids, device)
    test_results.to_csv("predictions/bart/etpc-paraphrase-detection-test-output.csv", index=False)


if __name__ == "__main__":
    args = get_args()

    name = "optim_seed"
    print(name)
    x = 1
    
    print("Start: ", x)
    seed_everything(args.seed)
    finetune_paraphrase_detection(args, x, name)
