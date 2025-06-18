import random
import enum
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import date
from utils.utils import load_json, load_pkl, save_pkl, parse_args
from typing import List, Dict, Tuple, Optional, Any

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')

class GenerationMethod(enum.Enum):
  """
  Enumeration of generation methods.

  by_const - the generation will be in constant mode: min_perturb, max_perturb are integer (for example, min_perturb = 1; max_perturb = 7);
  by_percentage - the generation will be in percentage mode: min_perturb, max_perturb = [0..1] (for example, min_perturb = 0.1; max_perturb = 0.75);
  by_size - similar to the percentage mode, required parameter is generation_limits, where for each maximum length of sentences are specified min_perturb and max_perturb ([0..1]):
    generation_limits = [
      [50, 0.1, 0.9],
      [100, 0.3, 0.75],
      ...
    ]

    [length of sentence, %min_perturb, %max_perturb]
  """
  by_const = 'by_const'
  by_percentage = 'by_percentage'
  by_size = 'by_size'

class GlyphProcessor:
  """
  GlyphProcessor - a tool for data analysis and generation.
  :param glyphs_dict: a dictionary where the key is a correct glyph, the value is a list of its homoglyphs -> {'a' : ['а', ...]};
  :type  glyphs_dict: Dict[str, List[str]];
  """
  def __init__(self, glyphs_dict: Dict[str, List[str]]):
    self.glyphs_dict = glyphs_dict
    
  def _get_index(self, lst, targets):
    indices = []
    for target in targets:
      indices += [i for i, letter in enumerate(lst) if letter == target]
    return indices
    
  def analyze_sentences(self, sentences: List[str], available_letters: List[str]) -> Tuple[pd.DataFrame, float, float]:
    """
    Analyzes input sentences.

    :param sentences: list of sentences to analyze;
    :type  sentences: List[str];
    :param available_letters: list of letters (glyphs) for which statistics will be calculated;
    :type  available_letters: List[str];
    :return: pandas DataFrame with main statistics, mean length of words, mean length of sentences;
    :rtype: Tuple[pd.DataFrame, float, float].
    """
    stop_words = ['.', ',', ';', ':', '?', '!', "'", '``', '`', '(', ')', '<', '>', '=', '{', '}', '/', '|', '~', '^', '%', '+', '-', '$']
    mean_word_length = []
    mean_sentence_length = []
    char_sum = 0

    alphabet = {}

    for char in available_letters:
      alphabet[char] = 0

    pbar = tqdm(sentences)

    for sentence in pbar:
      words = word_tokenize(sentence)
      words = [word for word in words if word not in stop_words]
      words_length = [len(word) for word in words]

      mean_word_length.append(np.mean(words_length))
      mean_sentence_length.append(len(sentence))

      words = "".join(words)
      for char in words:
        if char in available_letters:
          alphabet[char] += 1
          char_sum += 1

    print()
    print(f'Mean word`s length: {np.mean(mean_word_length)}')
    print(f'Mean sentence`s length: {np.mean(mean_sentence_length)}')
    print(f'Count of letters: {char_sum}')
    print('-'*20)

    letters = np.array(list(alphabet.keys()))
    count = np.array(list(alphabet.values()), dtype='float32')

    data = {
      'letters': letters,
      'count': count,
      'frequency': count/char_sum,
      'freq%': (count/char_sum)*100,
      'priority': 1 - (count/char_sum)
    }

    df = pd.DataFrame(data)

    return df, mean_word_length, mean_sentence_length
    
  def perturbate(
      self, 
      sentences: List[str], 
      method: str, 
      priority: Dict[str, float], 
      min_perturb: int = 1, 
      max_perturb: int = 1, 
      limits: Optional[List[Any]] = None
  ) -> List[List[str]]:
    """
    Generates dataset (makes perturbations in sentences by replacing glyphs with their homoglyphs).

    :param sentences: list of sentences to perturbate;
    :type  sentences: List[str];
    :param method: generation method (check enum GenerationMethod);
    :type  method: str;
    :param priority: a dictionary where the key is a correct glyph, the value is its priority (reverse frequency);
    :type  priority: Dict[str, float];
    :param min_perturb: a minimum count of perturbations in a single sentence (default = 1);
    :type  min_perturb: int;
    :param max_perturb: a maximum count of perturbations in a single sentence (default = 1);
    :type  max_perturb: int;
    :param limits: generation limits ([sentences_length, %min_perturb, %max_perturb]) if you choose GenerationMethod.by_size (default = None);
    :type  limits: Optional[List[Any]];
    :return: dataset with perturbations in sentences;
    :rtype: List[List[str, str]].
    """
    glyphs_list = list(self.glyphs_dict.keys())
    order = dict(zip(glyphs_list, np.zeros(len(glyphs_list), dtype=np.int32)))

    data = []
    target = list(self.glyphs_dict.keys())
    pbar = tqdm(sentences)
    for sentence in pbar:
      w = list(sentence)
      list_index = self._get_index(w, target) 

      if method == GenerationMethod.by_const:
        if len(list_index) >= min_perturb:
          max_perturb = min(len(list_index), max_perturb)
        else:
          max_perturb = len(list_index)
          min_perturb = 1

      elif method == GenerationMethod.by_percentage:
        min_perturb = max(round(len(w)*min_perturb), 1)
        max_perturb = round(len(w)*max_perturb)

      elif method == GenerationMethod.by_size:
        for limit in limits:
          if len(w) <= limit[0]:
            min_perturb = max(round(len(w)*limit[1]), 1)
            max_perturb = round(len(w)*limit[2])
            break
          else:
            min_perturb = 1
            max_perturb = round(len(w)*0.055)   

      for perturb in range(min_perturb, max_perturb + 1):       
        char_priority = [(priority[sentence[idx]], random.random(), idx) for idx in list_index]
        index = [x[2] for x in sorted(char_priority)[-perturb:]]        
        for i in index:
          w[i] = self.glyphs_dict[sentence[i]][order[sentence[i]]]
          order[sentence[i]] = (order[sentence[i]] + 1) % len(self.glyphs_dict[sentence[i]])
        data.append(["".join(w), sentence])
        w = list(sentence)     

    return data

if __name__ == "__main__":
  args = parse_args("Process the data: analyze and corrupt")
  config = load_json(args.config_path)
  
  glyphs_dict = load_pkl(config['homoglyphs_dict_path'])
  sentences = load_pkl(config['preprocessed_data_path'])
  generation_method = GenerationMethod(config['generation_method'])
  generation_limits = config['generation_limits']
  processor = GlyphProcessor(glyphs_dict)
  available_glyphs = list(processor.glyphs_dict.keys())

  print('Analyzing the data before it gets corrupted...')
  analyze, _mlw, _mls = processor.analyze_sentences(sentences, available_glyphs)
  print(analyze.head())
  print()

  print('Corrupting the data...')
  priority = dict(analyze.set_index('letters')['priority'].to_dict().items())
  dataset = processor.perturbate(sentences, generation_method, priority, limits=generation_limits)
  size_ds = len(dataset)

  print('Splitting the dataset...')
  train_size = int(size_ds*config["train_size"])
  random.shuffle(dataset)
  print(f'Length of the dataset: {size_ds}, train - {train_size}, test - {size_ds - train_size}')
  train_dataset_path = f'{config["save_train_dataset_dir"]}{config["dataset_name"]}_train_{date.today()}.pkl'
  test_dataset_path = f'{config["save_test_dataset_dir"]}{config["dataset_name"]}_test_{date.today()}.pkl'
  print('Saving the datasets...')
  save_pkl(dataset[:train_size], train_dataset_path)
  save_pkl(dataset[train_size:], test_dataset_path)
  print('Done!')
