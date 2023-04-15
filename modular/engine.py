"""
Contains functions for training and testing a PyTorch model.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, cast

import numpy as np
import torch
from tqdm.auto import tqdm
from .utils import save_model


def train_step(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (inputs, labels) in tqdm(enumerate(dataloader)):
        # 0. Send data to target device
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. Forward pass
        outputs = model(inputs)

        # 2. Calculate  and accumulate loss
        loss = criterion(outputs, labels)
        train_loss += loss.item() * inputs.size(0)

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Update the lr base on the predefined scheduler
        if scheduler:
            scheduler.step()

        # 7. Calculate and accumulate accuracy metric across all batches
        preds = torch.argmax(outputs, dim=1)
        train_acc += torch.sum(preds == labels).item()

    train_loss = train_loss / len(dataloader.dataset)
    train_acc = train_acc / len(dataloader.dataset)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (inputs, labels) in tqdm(enumerate(dataloader)):
            # 0. Send data to target device
            inputs, labels = inputs.to(device), labels.to(device)

            # 1. Forward pass
            outputs = model(inputs)

            # 2. Calculate and accumulate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            # 3. Calculate and accumulate accuracy metric across all batches
            preds = outputs.argmax(dim=1)
            test_acc += torch.sum(preds == labels).item()

    test_loss = test_loss / len(dataloader.dataset)
    test_acc = test_acc / len(dataloader.dataset)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: torch.device,
          patience: int = 5,
          scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          save_as: Optional[Path] = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]}
    For example if training for epochs=2:
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]}
    """

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Makes sure model on target device
    model.to(device)

    # Init loss
    valid_loss_min = np.Inf

    # Init early stop
    early_stop_count = 0

    # Is save checkpoint required?
    is_save_required = save_as is not None

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           device=device,
                                           scheduler=scheduler)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        criterion=criterion,
                                        device=device)

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        # Compute if network learned comparing the test loss
        network_learned = test_loss < valid_loss_min

        if network_learned:
            valid_loss_min = test_loss
            early_stop_count = 0

            # Save network checkpoint
            if is_save_required:
                data_dict = {
                    'train_loss': results["train_loss"],
                    'train_acc': results["train_acc"],
                    'test_loss': results["test_loss"],
                    'test_acc': results["test_acc"],
                    'epoch': epoch,
                    'optimizer': optimizer.state_dict(),
                    'model': model.state_dict()
                }
                save_model(data_dict, cast(Path, save_as))
        else:
            early_stop_count += 1
            if early_stop_count >= patience:
                print(f'Early stopping after {epoch} epochs')
                break

    # Returns train history
    return results
