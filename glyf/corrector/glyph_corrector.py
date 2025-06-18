from tqdm import tqdm
import torch
import numpy as np
from transformers import T5ForConditionalGeneration, AutoTokenizer
from utils.utils import load_pkl, calculate_accruracy, calculate_levenshtein_ratio
from logs.logger import Logger
from typing import Union, List, Dict, Tuple, Optional

class GlyphCorrector:
    """
    Inference class for trained model.

    __init__ params:
    :param model_path: path to model;
    :type  model_path: string;
    :param glyphs_path: path to dictionary of homoglyphs;
    :type  glyphs_path: Dict[str, List[str]];
    :param prefix: additional prompt for model (for example, 'fix homoglyphs: '), some models need it;
    :type  prefix: str;
    :param device: device on which the model will be located (cpu/gpu);
    :type  device: str or torch.device;
    """
    def __init__(self, model_path: str, glyphs_path: Dict[str, List[str]], prefix: Optional[str], device: Union[str, torch.device]):
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

    def correct(self, sentence: str) -> str:
        """
        Corrects a single input sentence.

        :param x: input sentence;
        :type  x: str;
        :return: corrected sentence;
        :rtype: str.
        """
        return self.batch_correct([sentence], batch_size=1)[-1][0]
    
    def batch_correct(self, sentences: List[str], batch_size: int) -> List[List[str]]:
        """
        Corrects input list of sentences.

        :param sentences: input list of sentences;
        :type  sentences: List[str];
        :param batch_size: size of subsample of input sentences;
        :type  batch_size: int;
        :return: corrected sentences;
        :rtype: List[List[str]];
        """
        batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        result = []
        batches_bar = tqdm(batches)
        for batch in batches_bar:
            batch = [self.prefix + x for x in batch]
            with torch.inference_mode():
                encodings = self.tokenizer(batch, return_tensors='pt', padding='longest').to(self.device)
                generated_tokens = self.model.generate(**encodings)
                result.append(self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))
        return result
    
    def evaluate(self, dataset: List[List[str]], batch_size: int = 32, logs_path: str ="") -> Tuple[float, float]:
        """
        Evaluate the model on dataset.

        :param dataset: data ([[X, y], ...], X - attacked sentence, y - corrected sentence);
        :type  dataset: List[List[str]];
        :param batch_size: size of subsample of input sentences (default = 32);
        :type  batch_size: int;
        :param logs_path: path to file for logging (for example, 'logs.log');
        :type  logs_path: str;
        :return: calculated metrics on the dataset (accuracy and levenshtein ratio);
        :rtype: Tuple[float, float].
        """
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