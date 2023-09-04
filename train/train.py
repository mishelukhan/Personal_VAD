import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def train_loop(model, criterion, optimizer, scheduler, path_checkpoint, train_dl, 
               val_dl, test_dl, device, epochs=10, need_visualisation=False):

    loss_train = []
    loss_val = []
    loss_test = []
    loss_train_per_epoch = []
    loss_val_per_epoch = []
    loss_test_per_epoch = []

    accuracies_train = []
    accuracies_val = []
    accuracies_test = []
    accuracies_train_per_epoch = []
    accuracies_val_per_epoch = []
    accuracies_test_per_epoch = []

    f1_train = []
    f1_val = []
    f1_test = []
    f1_train_per_epoch = []
    f1_val_per_epoch = []
    f1_test_per_epoch = []

    map_train = []
    map_val = []
    map_test = []
    map_train_per_epoch = []
    map_val_per_epoch = []
    map_test_per_epoch = []

    best_val_loss = float('inf')

    for epoch in tqdm(range(epochs)):

        'Train phase'
        model.train()
        for iter, batch in enumerate(tqdm(train_dl)):
            optimizer.zero_grad()
            input = batch[0].to(device)
            labels = batch[1].to(device).flatten()
            out = model(input, [input.shape[1]]*input.shape[0]).flatten(start_dim=0, end_dim=1)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            prediction = out.argmax(dim=1).int().detach().cpu().numpy().flatten()
            probs = out.detach().cpu().numpy().flatten()
            targets = labels.int().detach().cpu().numpy().flatten()

            loss_train.append(loss.item())

            acc = accuracy_score(targets, prediction)
            accuracies_train.append(acc)

            f1 = f1_score(targets, prediction, average='macro')
            f1_train.append(f1)

            map = average_precision_score(targets, probs, average='micro')
            map_train.append(map)
            if iter % 200 == 0:
                print(f'accuracy: {acc}, loss: {loss.item()}, f1-score: {f1}, map: {map}')

        scheduler.step()

        print(f'Mean train loss per epoch {epoch + 1}: {sum(loss_train) / len(loss_train)}')
        print(f'Mean train accuracy per epoch {epoch + 1}: {sum(accuracies_train) / len(accuracies_train)}')
        print(f'Mean train f1-score per epoch {epoch + 1}: {sum(f1_train) / len(f1_train)}')
        print(f'Mean train map-score per epoch {epoch + 1}: {sum(map_train) / len(map_train)}')
        loss_train_per_epoch.append(sum(loss_train) / len(loss_train))
        accuracies_train_per_epoch.append(sum(accuracies_train) / len(accuracies_train))
        f1_train_per_epoch.append(sum(f1_train) / len(f1_train))
        map_train_per_epoch.append(sum(map_train) / len(map_train))

        'Validation phase'
        model.eval()
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(val_dl)):
                input = batch[0].to(device)
                labels = batch[1].to(device).flatten()
                out = model(input).flatten(start_dim=0, end_dim=1)
                loss = criterion(out, labels)

                prediction = out.argmax(dim=1).int().detach().cpu().numpy().flatten()
                probs = out.detach().cpu().numpy().flatten()
                targets = labels.int().detach().cpu().numpy().flatten()

                loss_val.append(loss.item())

                acc = accuracy_score(targets, prediction)
                accuracies_val.append(acc)

                f1 = f1_score(targets, prediction, average='macro')
                f1_val.append(f1)

                map = average_precision_score(targets, probs, average='micro')
                map_val.append(map)
                if iter % 100 == 0:
                    print(f'accuracy: {acc}, loss: {loss.item()}, f1-score: {f1}, map: {map}')

        print(f'Mean val loss per epoch {epoch + 1}: {sum(loss_val) / len(loss_val)}')
        print(f'Mean val accuracy per epoch {epoch + 1}: {sum(accuracies_val) / len(accuracies_val)}')
        print(f'Mean val f1-score per epoch {epoch + 1}: {sum(f1_val) / len(f1_val)}')
        print(f'Mean val map-score per epoch {epoch + 1}: {sum(map_val) / len(map_val)}')
        loss_val_per_epoch.append(sum(loss_val) / len(loss_val))
        accuracies_val_per_epoch.append(sum(accuracies_val) / len(accuracies_val))
        f1_val_per_epoch.append(sum(f1_val) / len(f1_val))
        map_val_per_epoch.append(sum(map_val) / len(map_val))
        
        if loss_val_per_epoch[-1] < best_val_loss:
            checkpoint = {
                'epochs': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(checkpoint, path_checkpoint)

        'Test phase'
        model.eval()
        with torch.no_grad():
            for iter, batch in enumerate(tqdm(test_dl)):
                input = batch[0].to(device)
                labels = batch[1].to(device).flatten()
                out = model(input).flatten(start_dim=0, end_dim=1)
                loss = criterion(out, labels)

                prediction = out.argmax(dim=1).int().detach().cpu().numpy().flatten()
                probs = out.detach().cpu().numpy().flatten()
                targets = labels.int().detach().cpu().numpy().flatten()

                loss_test.append(loss.item())

                acc = accuracy_score(targets, prediction)
                accuracies_test.append(acc)

                f1 = f1_score(targets, prediction, average='macro')
                f1_test.append(f1)

                map = average_precision_score(targets, probs, average='micro')
                map_test.append(map)
                if iter % 100 == 0:
                    print(f'accuracy: {acc}, loss: {loss.item()}, f1-score: {f1}, map: {map}')

        print(f'Mean test loss per epoch: {sum(loss_test) / len(loss_test)}')
        print(f'Mean test accuracy per epoch: {sum(accuracies_test) / len(accuracies_test)}')
        print(f'Mean test f1-score per epoch: {sum(f1_test) / len(f1_test)}')
        print(f'Mean test map-score per epoch: {sum(map_test) / len(map_test)}')
        loss_test_per_epoch.append(sum(loss_test) / len(loss_test))
        accuracies_test_per_epoch.append(sum(accuracies_test) / len(accuracies_test))
        f1_test_per_epoch.append(sum(f1_test) / len(f1_test))
        map_test_per_epoch.append(sum(map_test) / len(map_test))

    'Plot phase'
    if need_visualisation:
        f, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)
        ax1.plot(np.arange(1, epochs + 1), loss_train_per_epoch, 'r')
        ax1.plot(np.arange(1, epochs + 1), loss_val_per_epoch, 'b')
        ax1.grid(True)
        ax1.set_title('Losses')
        ax1.legend(['train', 'test'])

        ax2.plot(np.arange(1, epochs + 1), accuracies_train_per_epoch, 'r')
        ax2.plot(np.arange(1, epochs + 1), accuracies_val_per_epoch, 'b')
        ax2.grid(True)
        ax2.set_title('Accuracy')
        ax2.legend(['train', 'test'])

        ax3.plot(np.arange(1, epochs + 1), f1_train_per_epoch, 'r')
        ax3.plot(np.arange(1, epochs + 1), f1_val_per_epoch, 'b')
        ax3.grid(True)
        ax3.set_title('F1-score')
        ax3.legend(['train', 'test'])

        ax4.plot(np.arange(1, epochs + 1), map_train_per_epoch, 'r')
        ax4.plot(np.arange(1, epochs + 1), map_val_per_epoch, 'b')
        ax4.grid(True)
        ax4.set_title('MAP-score')
        ax4.legend(['train', 'test'])

        plt.show()