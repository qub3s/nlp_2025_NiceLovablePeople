# This class has been adopted from the EarlyStopping class by Georg

import numpy as np
import torch

class PGEarlyStopping:
    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.bleu_score_min = np.inf
        self.delta = delta
        self.model_checkpoint_path = checkpoint_path
        self.model = None

    def __call__(self, bleu_score, model, epoch):
        score = bleu_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(bleu_score, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                torch.save(self.model, f"{self.model_checkpoint_path}{epoch}.pt")
                if self.verbose:
                    print(f'Penalised Bleu Score Decreased ({self.bleu_score_min:.6f} --> {bleu_score:.6f}).  Saving model in epoch ', epoch)
        else:
            self.best_score = score
            self.save_checkpoint(bleu_score, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, bleu_score, model):
        self.model = model.state_dict()
        self.bleu_score_min = bleu_score