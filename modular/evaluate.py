"""
Script to evaluate a model
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def get_transpose(img: torch.Tensor, trans):
    if trans >= 4:
        img = img.transpose(2, 3)
    if trans % 4 == 0:
        return img
    elif trans % 4 == 1:
        return img.flip(2)
    elif trans % 4 == 2:
        return img.flip(3)
    elif trans % 4 == 3:
        return img.flip(2).flip(3)


@torch.inference_mode()
def val_step(model: nn.Module,
             loader: torch.utils.data.DataLoader,
             device: torch.device,
             out_dim: int,
             n_test: int = 1):

    model.eval()

    LOGITS = []
    PROBS = []
    LABELS = []

    for inputs in tqdm(loader):

        inputs = inputs.to(device)
        logits = torch.zeros((inputs.shape[0], out_dim)).to(device)
        probs = torch.zeros((inputs.shape[0], out_dim)).to(device)

        # Multiple test
        for test in range(n_test):
            test_logits = model(get_transpose(inputs, test))
            logits += test_logits
            probs += torch.softmax(test_logits, dim=1)

        logits /= n_test
        probs /= n_test
        labels = torch.argmax(probs, dim=1)

        LOGITS.append(logits.detach().cpu())
        PROBS.append(probs.detach().cpu())
        LABELS.append(labels.detach().cpu())

    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    LABELS = torch.cat(LABELS).numpy()

    return LOGITS, PROBS, LABELS
