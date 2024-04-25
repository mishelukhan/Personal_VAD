import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, confusion_matrix
from tqdm.notebook import tqdm
import numpy as np
import sys

def training_denoiser_lstm(model, criterion_near, criterion_far, optimizer, path_checkpoint, 
                           train_dl, val_dl, device, epochs=10):

    best_val_loss = float('inf')
    loss_list = []
    val_loss_list = []
    epochs_list = []

    for epoch in range(epochs):
        loss_train = 0
        loss_val = 0
    
        accuracies_train = []
        accuracies_val = []
    
        f1_train = []
        f1_val = []
    
        map_train = []
        map_val = []
        model.train()
        for iteration, batch_features in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            
            input_map = batch_features[0].to(device).to(torch.float32)
            logits = batch_features[1].to(device)
            target_map = batch_features[2].to(device)
            fields = batch_features[3].to(device)
            output_map, _ = model(input_map)
            logits[..., 2] = logits[..., 2] + output_map[..., 0]
            logits_near = logits[fields == 1].flatten(start_dim=0, end_dim=-2)
            logits_far = logits[fields == 0].flatten(start_dim=0, end_dim=-2)
            target_map_near = target_map[fields == 1].flatten()
            target_map_far = target_map[fields == 0].flatten()
            train_loss = criterion_near(logits_near, target_map_near) + criterion_far(logits_far, target_map_far)

            # compute accumulated gradients
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss_train += train_loss.item()

            prediction = logits.argmax(dim=2).int().detach().cpu().numpy().flatten()
            probs = logits.detach().cpu().flatten(start_dim=0, end_dim=-2).numpy()
            targets = target_map.detach().cpu().numpy().flatten()
            targets_oh = np.eye(3)[targets]

            acc = accuracy_score(targets, prediction)
            accuracies_train.append(acc)

            f1 = f1_score(targets, prediction, average='micro')
            f1_train.append(f1)
            map_score = average_precision_score(targets_oh, probs, average='micro')
            map_train.append(map_score)
            if iteration % 50 == 0:
                print(f'accuracy: {acc}, loss: {train_loss.item()}, f1-score: {f1}, map: {map_score}')

        # compute the epoch training loss
        loss_train = loss_train / len(train_dl)
        print(f'Mean train loss per epoch {epoch + 1}: {loss_train}')
        print(f'Mean train accuracy per epoch {epoch + 1}: {sum(accuracies_train) / len(accuracies_train)}')
        print(f'Mean train f1-score per epoch {epoch + 1}: {sum(f1_train) / len(f1_train)}')
        print(f'Mean train map-score per epoch {epoch + 1}: {sum(map_train) / len(map_train)}')
        loss_train = 0

        # compute the validation loss
        model.eval()
        with torch.no_grad():
            for iteration, batch_features in enumerate(tqdm(val_dl)):
                input_map = batch_features[0].to(device).to(torch.float32)
                logits = batch_features[1].to(device)
                target_map = batch_features[2].to(device)
                fields = batch_features[3].to(device)
                output_map, _ = model(input_map)
                logits[..., 2] = logits[..., 2] + output_map[..., 0]
                logits_near = logits[fields == 1].flatten(start_dim=0, end_dim=-2)
                logits_far = logits[fields == 0].flatten(start_dim=0, end_dim=-2)
                target_map_near = target_map[fields == 1].flatten()
                target_map_far = target_map[fields == 0].flatten()
                val_loss = criterion_near(logits_near, target_map_near) + criterion_far(logits_far, target_map_far)
    
                # add the mini-batch training loss to epoch loss
                loss_val += val_loss.item()

                prediction = logits.argmax(dim=2).int().detach().cpu().numpy().flatten()
                probs = logits.detach().cpu().flatten(start_dim=0, end_dim=-2).numpy()
                targets = target_map.detach().cpu().numpy().flatten()
                targets_oh = np.eye(3)[targets]
    
                acc = accuracy_score(targets, prediction)
                accuracies_val.append(acc)
    
                f1 = f1_score(targets, prediction, average='micro')
                f1_val.append(f1)
    
                map_score = average_precision_score(targets_oh, probs, average='micro')
                map_val.append(map_score)
    
                if iteration % 50 == 0:
                    print(f'Test: accuracy: {acc}, loss: {val_loss.item()}, f1-score: {f1}, map: {map_score}')

        # compute the epoch validation loss
        loss_val = loss_val / len(val_dl)
        print(f'Mean test loss per epoch {epoch + 1}: {loss_val}')
        print(f'Mean test accuracy per epoch {epoch + 1}: {sum(accuracies_val) / len(accuracies_val)}')
        print(f'Mean test f1-score per epoch {epoch + 1}: {sum(f1_val) / len(f1_val)}')
        print(f'Mean test map-score per epoch {epoch + 1}: {sum(map_val) / len(map_val)}')
        if loss_val < best_val_loss:
            checkpoint = {
                'epochs': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }
            torch.save(checkpoint, path_checkpoint)
        loss_val = 0
