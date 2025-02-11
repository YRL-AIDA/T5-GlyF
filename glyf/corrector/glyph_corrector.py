from tqdm import tqdm
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils.utils import load_pkl, calculate_accruracy, calculate_levenshtein_ratio
from logs.logger import Logger

class GlyphCorrector:
    def __init__(self, model_path, glyphs_path, prefix, device):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.glyphs_dict = load_pkl(glyphs_path)
        self.prefix = prefix
        self.device = device

        tokens = list(set().union(*self.glyphs_dict.values()))
        tokens = set(tokens) - set(self.tokenizer.vocab.keys())
        self.tokenizer.add_tokens(list(tokens))
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model.to(self.device)
        self.model.eval()

    def correct(self, x):
        return self.batch_correct([x], batch_size=1)[-1][0]
    
    def batch_correct(self, x_batch, batch_size):
        batches = [x_batch[i:i + batch_size] for i in range(0, len(x_batch), batch_size)]
        result = []
        batches_bar = tqdm(batches)
        for batch in batches_bar:
            batch = [self.prefix + x for x in batch]
            with torch.inference_mode():
                encodings = self.tokenizer(batch, return_tensors='pt', padding='longest').to(self.device)
                generated_tokens = self.model.generate(**encodings)
                result.append(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
        return result
    
    def evaluate(self, dataset, batch_size=32, logs_path=''):
        y_true = [x[1] for x in dataset]
        x = [x[0] for x in dataset]

        y_pred = sum(self.batch_correct(x, batch_size=batch_size), [])
        accuracy = calculate_accruracy(y_pred, y_true)
        l_ratio = calculate_levenshtein_ratio(y_pred, y_true)

        if logs_path != '':
            logger = Logger(logs_path)
            msg = f'Size of dataset: {len(dataset)}; batch_size: {batch_size}'
            logger.info("test-params", msg)
            msg = f'Accuracy: {accuracy}; l-ratio: {l_ratio}'
            logger.info("testing", msg)

        return accuracy, l_ratio