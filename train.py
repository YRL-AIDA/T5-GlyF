from datetime import date
import numpy as np
from tqdm import tqdm
import copy

import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer

from glyf.data_processing.dataset import HomoglyphDataset
from logs.logger import Logger
from utils.utils import parse_args, load_json, load_pkl, plot_losses

def fit(model, tokenizer, train, val, prefix, device, optim, epochs, learning_rate, batch_size, early_stopping_patience, logs_path):

    model.to(device)
    optimizer = optim(model.parameters(), lr=learning_rate)
    
    logger = Logger(logs_path)
    logger.info("train", f"Starting train the model on ({device})")
    logger.info("train-params", f"Size of dataset: {len(train)}; batch_size: {batch_size}, learning rate: {learning_rate}")

    train = DataLoader(train, batch_size=batch_size, shuffle=True)
    val = DataLoader(val, batch_size=batch_size, shuffle=False)

    train_loss = []
    val_loss = []
    best_val_loss = float('inf')
    best_epoch = 0

    space = '-'*63

    for epoch in range(epochs):
        mean_loss = 0
        batch_n = 0
        model.train(True)

        pbar_train = tqdm(train, desc=f'Epoch: {epoch + 1}/{epochs}')

        for x_train, y_train in pbar_train:
            x = tokenizer([prefix + x for x in x_train], return_tensors='pt', padding='longest').to(device)
            y = tokenizer(y_train, return_tensors='pt', padding='longest').to(device)

            loss = model(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                labels=y.input_ids,
                return_dict=True
            ).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mean_loss += loss.item()
            batch_n += 1
            pbar_train.set_postfix(loss_train=round(mean_loss/batch_n, 6), refresh=True)

        mean_loss /= batch_n
        train_loss.append(mean_loss)

        model.eval()
        mean_loss = 0
        batch_n = 0
        with torch.no_grad():
            for x_train_val, y_train_val in val:
                x = tokenizer([prefix + x for x in x_train_val], return_tensors='pt', padding='longest').to(device)
                y = tokenizer(y_train_val, return_tensors='pt', padding='longest').to(device)

                loss = model(
                    input_ids=x.input_ids,
                    attention_mask=x.attention_mask,
                    labels=y.input_ids,
                    return_dict=True
                ).loss

                mean_loss += loss.item()
                batch_n += 1

        mean_loss /= batch_n
        val_loss.append(mean_loss)
        logger.info("train", f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss[-1]}; valid_loss: {val_loss[-1]}")

        print(f'{space}loss_valid={round(mean_loss, 6)}')

        if mean_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = mean_loss
            best_model = copy.deepcopy(model)
            print('^'*32 + 'new best model' + '^'*32)
            print()
        elif epoch - best_epoch > early_stopping_patience:
            print(f'Model has not improved in the last {early_stopping_patience} epochs. Break.')
            break

    return best_model, train_loss, val_loss

if __name__ == "__main__":
    args = parse_args("Training the model")
    config = load_json(args.config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = config['model_path']
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    glyphs_dict = load_pkl(config['homoglyphs_dict_path'])

    tokens = list(set().union(*glyphs_dict.values()))
    tokens = set(tokens) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(tokens))
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    dataset_path = config['train_dataset_path']
    dataset = load_pkl(dataset_path)
    size_ds = len(dataset)
    train_size = train_size = int(size_ds*config["train_size"])

    X = [x[0] for x in dataset]
    y = [x[1] for x in dataset]

    x_train = X[:train_size]
    y_train = y[:train_size]

    x_valid = X[train_size:]
    y_valid = y[train_size:]

    train = HomoglyphDataset(x_train, y_train)
    val = HomoglyphDataset(x_valid, y_valid)

    prefix = config['generation_prefix']
    epochs = config['number_of_epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    early_stop = config['early_stopping_patience']
    logs_path = config['logs_path']

    fit_params = {
        'model': model,
        'tokenizer': tokenizer,
        'train': train,
        'val': val,
        'prefix': prefix,
        'device': device,
        'optim': torch.optim.Adam,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'early_stopping_patience': early_stop,
        'logs_path': logs_path
    }

    best_model, train_loss, val_loss = fit(**fit_params)
    print('\nSaving the best model...')

    save_model_dir = f"{config['save_model_dir']}/model_{date.today()}/"

    best_model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)

    save_chart_path = f"{config['save_chart_dir']}losses_{date.today()}.png"
    plot_losses(train_loss, val_loss, save_chart_path)





