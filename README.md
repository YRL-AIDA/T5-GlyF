# T5-GlyF 
**T5-GlyF** (**Gly**ph **F**ixing) — a method to correct text data attacked with homoglyphs using a pre-trained T5 model by the SberDevices team as part of the SAGE project.
## Installation
To get started, you need to install the requirements
```commandline
git clone https://github.com/YRL-AIDA/T5-GlyF.git
cd T5-GlyF
pip install -r requirements.txt
```
## Interaction with the model
### Training
To start training the model, simply enter the following command
```commandline
python train.py --config_path configs/train_config.json
```
### Testing
To start testing the model
```commandline
python test.py --config_path configs/test_config.json
```