import itertools
import sys
from argparse import ArgumentParser
from typing import Tuple, Callable, Dict
from os import path
import torch
from numpy import ndarray
from torch import Tensor
from torch.nn import NLLLoss, CrossEntropyLoss
from torch.optim import Optimizer, Adam
from torch.utils.data.dataloader import DataLoader
import numpy as np
from gcommand_dataset import GCommandLoader
from model import AudioCommandClassifier


def train_model(model: AudioCommandClassifier, optimizer: Optimizer, loss_function: Callable, train_loader: DataLoader,
                epochs: int) -> Tuple[ndarray, ndarray]:
    loss_list = []
    accuracy_list = []
    for i in range(epochs):
        running_loss = 0.0
        total = 0
        correct = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            predictions = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        accuracy_list.append(100 * correct / total)
        loss_list.append(running_loss / total)
        print(f'Epoch: {i + 1} --> loss: {loss_list[-1]:.3f} accuracy: {accuracy_list[-1]:.3f}%')
    return np.array(loss_list), np.array(accuracy_list)


def test(validation_loader: DataLoader, model: AudioCommandClassifier, loss_function: Callable) -> Tuple[float, float]:
    total = 0
    correct = 0
    loss = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            loss += loss_function(outputs, labels.type(torch.long)).sum().item()
            predictions = torch.argmax(outputs.data, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return loss / total, 100 * correct / total


def predict(model: AudioCommandClassifier, pred_loader: DataLoader) -> Tensor:
    model.eval()
    with torch.no_grad():
        predictions = []
        for inputs, _ in pred_loader:
            outputs = model(inputs)
            predictions.extend(list(torch.argmax(outputs.data, 1)))
        return np.array(predictions)


def main(batch_size: int, lr: float, epochs: int, loss_function: Callable, base_folder: str):
    model = AudioCommandClassifier(30)
    optimizer = Adam(model.parameters(), lr=lr)

    dataset = GCommandLoader(path.join(base_folder, 'train'))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    loss, accuracy = train_model(model, optimizer, loss_function, train_loader, epochs)
    print(f'Averages --> loss: {np.average(loss[-5:]):.3f}, accuracy: {np.average(accuracy[-5:]):.3f}')

    validation_dataset = GCommandLoader(path.join(base_folder, 'valid'))
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    loss, accuracy = test(validation_loader, model, loss_function)
    print(f'Validation --> loss: {loss:.3f}, accuracy: {accuracy:.3f}')

    test_dataset = GCommandLoader(path.join(base_folder, 'test'), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=2)
    preds = predict(model, test_loader)
    return loss, accuracy, preds


def are_text_files_equal(file_name_1, file_name_2):
    total = 0
    correct = 0
    with open(file_name_1, 'r') as file1:
        with open(file_name_2, 'r') as file2:
            lines_1 = file1.readlines()
            lines_2 = file2.readlines()
            for line1, line2 in zip(lines_1, lines_2):
                if line1.strip() == line2.strip():
                    correct += 1
                total += 1
    return 100 * correct / total


def hyper_parameters_grid_search(base_folder: str, file_base_name: str, num_to_class: Dict):
    batches = [32, 64, 128, 256]
    lrs = [0.01, 0.001, 0.0001]
    epochs = [10, 15, 20]
    loss_funcs = [NLLLoss(), CrossEntropyLoss()]
    best_values = {'loss': 0, 'accuracy': 0, 'predictions': None, 'index': 0}
    for i, (batch_size, lr, epochs, loss_func) in enumerate(itertools.product(batches, lrs, epochs, loss_funcs)):
        print(f'Run #{i + 1} --> batch: {batch_size}, lr: {lr}, epochs: {epochs}, loss_func: {loss_func}')
        loss, accuracy, preds = main(batch_size, lr, epochs, loss_func, base_folder)
        write_predictions_to_file(preds, f'{file_base_name}_{i + 1}', num_to_class)
        if accuracy > best_values['accuracy']:
            best_values['index'] = i
            best_values['loss'] = loss
            best_values['accuracy'] = accuracy
            best_values['predictions'] = preds
    print(f"Best: run #{best_values['index']} --> loss: {best_values['loss']}, accuracy: {best_values['accuracy']}")
    return best_values


def write_predictions_to_file(predictions: ndarray, file_name: str, num_to_class: Dict):
    with open(file_name, 'w') as file:
        for i, pred in enumerate(predictions):
            file.write(f'{i}.wav,{num_to_class[pred]}\n')


if __name__ == '__main__':
    parser = ArgumentParser(description="Runs the Audio Command Classifier neural network")
    parser.add_argument('data_folder', type=str, help='The path to the gcommands data foler')
    parser.add_argument('-c', '--check_hyper_parameters', action='store_true', default=False,
                        help='Use this flag to run grid search for hyper-parameters')
    args = parser.parse_args(sys.argv[1:])
    CLASSES = {'bed': 0, 'bird': 1, 'cat': 2, 'dog': 3, 'down': 4, 'eight': 5, 'five': 6, 'four': 7, 'go': 8,
               'happy': 9, 'house': 10, 'left': 11, 'marvin': 12, 'nine': 13, 'no': 14, 'off': 15, 'on': 16, 'one': 17,
               'right': 18, 'seven': 19, 'sheila': 20, 'six': 21, 'stop': 22, 'three': 23, 'tree': 24, 'two': 25,
               'up': 26, 'wow': 27, 'yes': 28, 'zero': 29}
    INVERT_CLASSES = {}
    for key, value in CLASSES.items():
        INVERT_CLASSES[value] = key
    FILE_NAME = 'test_y'
    DATA_FOLDER = args.data_folder
    if args.check_hyper_parameters:
        best = hyper_parameters_grid_search(DATA_FOLDER, FILE_NAME, INVERT_CLASSES)
        predictions = best['predictions']
    else:
        # these are the best hyper-params we found so we use those
        loss, accuracy, predictions = main(32, 0.001, 20, NLLLoss(), DATA_FOLDER)
    write_predictions_to_file(predictions, FILE_NAME, INVERT_CLASSES)
    print('Finished!')
