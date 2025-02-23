from glyf.corrector.glyph_corrector import GlyphCorrector
from utils.utils import parse_args, load_json, load_pkl
import torch

if __name__ == "__main__":
    args = parse_args("Testing the model")
    config = load_json(args.config_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_params = {
        'model_path': config['model_path'],
        'glyphs_path': config['homoglyphs_dict_path'],
        'prefix': config['generation_prefix'],
        'device': device
    }

    corrector = GlyphCorrector(**test_params)

    dataset_path = config['test_dataset_path']
    batch_size = config['test_batch_size']
    logs_path = config['logs_path']

    ds = load_pkl(dataset_path)

    acc, l_ratio = corrector.evaluate(ds, batch_size, logs_path)
    print(f'Testing is over.\nAccuracy: {acc}; levenshtein-ratio: {l_ratio}')


    
