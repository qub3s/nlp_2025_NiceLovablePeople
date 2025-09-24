# This class has been adopted from the EarlyStopping class by Georg Eckardt

import numpy as np
import torch

class PGEarlyStopping:
    def __init__(self, checkpoint_path, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.bleu_score_max = np.inf
        self.saved_epoch = -1
        self.delta = delta
        self.model_checkpoint_path = checkpoint_path
        self.model = None

    def __call__(self, bleu_score, model, epoch):
        score = bleu_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, epoch)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                torch.save(self.model, f"{self.model_checkpoint_path}{self.saved_epoch}.pt")
                if self.verbose:
                    print(f'Penalised Bleu Score Decreased ({self.bleu_score_max:.6f} --> {bleu_score:.6f}) in epoch {epoch}.  Saving model from epoch ', self.saved_epoch)
        else:
            self.best_score = score
            self.save_checkpoint(bleu_score, model, epoch)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, bleu_score, model, epoch):
        self.model = model.state_dict()
        self.bleu_score_max = bleu_score
        self.saved_epoch = epoch