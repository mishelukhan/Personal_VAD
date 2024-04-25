import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from tqdm.notebook import tqdm
import numpy as np
import sys


def train_loop(model, criterion, optimizer, path_checkpoint, train_dl, 
               val_dl, device, epochs=10):
    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs), position=0, file=sys.stdout):
        loss_train = []
        loss_val = []
    
        accuracies_train = []
        accuracies_val = []
    
        f1_train = []
        f1_val = []
    
        map_train = []
        map_val = []

        'Train phase'
        model.train()
        for iteration, batch in enumerate(tqdm(train_dl, position=0, file=sys.stdout)):
            optimizer.zero_grad()
            x_input = batch[0].to(device).to(torch.float)
            labels = batch[1].to(device).flatten()
            out = model(x_input)[0].flatten(start_dim=0, end_dim=1)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            prediction = out.argmax(dim=1).int().detach().cpu().numpy().flatten()
            probs = out.detach().cpu().numpy()
            targets = labels.int().detach().cpu().numpy().flatten()
            targets_oh = np.eye(3)[targets]

            loss_train.append(loss.item())

            acc = accuracy_score(targets, prediction)
            accuracies_train.append(acc)

            f1 = f1_score(targets, prediction, average='micro')
            f1_train.append(f1)

            map_score = average_precision_score(targets_oh, probs, average='micro')
            map_train.append(map_score)
            if iteration % 200 == 0:
                print(f'accuracy: {acc}, loss: {loss.item()}, f1-score: {f1}, map: {map_score}')

        print(f'Mean train loss per epoch {epoch + 1}: {sum(loss_train) / len(loss_train)}')
        print(f'Mean train accuracy per epoch {epoch + 1}: {sum(accuracies_train) / len(accuracies_train)}')
        print(f'Mean train f1-score per epoch {epoch + 1}: {sum(f1_train) / len(f1_train)}')
        print(f'Mean train map-score per epoch {epoch + 1}: {sum(map_train) / len(map_train)}')

        'Validation phase'
        model.eval()
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(val_dl, position=0, file=sys.stdout)):
                x_input = batch[0].to(device).to(torch.float)
                labels = batch[1].to(device).flatten()
                out = model(x_input)[0].flatten(start_dim=0, end_dim=1)
                loss = criterion(out, labels)

                prediction = out.argmax(dim=1).int().detach().cpu().numpy().flatten()
                probs = out.detach().cpu().numpy()
                targets = labels.int().detach().cpu().numpy().flatten()
                targets_oh = np.eye(3)[targets]

                loss_val.append(loss.item())

                acc = accuracy_score(targets, prediction)
                accuracies_val.append(acc)

                f1 = f1_score(targets, prediction, average='micro')
                f1_val.append(f1)

                map_score = average_precision_score(targets_oh, probs, average='micro')
                map_val.append(map_score)
                if iteration % 100 == 0:
                    print(f'accuracy: {acc}, loss: {loss.item()}, f1-score: {f1}, map: {map_score}')

        print(f'Mean val loss per epoch {epoch + 1}: {sum(loss_val) / len(loss_val)}')
        print(f'Mean val accuracy per epoch {epoch + 1}: {sum(accuracies_val) / len(accuracies_val)}')
        print(f'Mean val f1-score per epoch {epoch + 1}: {sum(f1_val) / len(f1_val)}')
        print(f'Mean val map-score per epoch {epoch + 1}: {sum(map_val) / len(map_val)}')

        mean_val_loss = sum(loss_val) / len(loss_val)
        if mean_val_loss < best_val_loss:
            checkpoint = {
                'epochs': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict()
            }
            torch.save(checkpoint, path_checkpoint)
