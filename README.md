# MoBERT
MoBERT: A Strongly Correlated Text-to-Motion Evaluation Metric
# Installation
This code is tested on Linux Ubuntu 22.04 and with python 3.9. 
Using Conda run the following to create a enviroment.
````
conda env create -f environment.yml
````
Download checkpoints from this drive and unzip them in the checkpoints/primary_evaluator/directory
````
https://drive.google.com/file/d/1gmljNRJKf_IujUIlcmCl9Q6mZI_Qceiv/view?usp=sharing
````
Download our evaluation dataset from drive and unzip them in the MotionDataset/folder.
````
https://drive.google.com/file/d/1obWxAH0kh3WaVbcfcetMhaxF38FqBmqp/view?usp=sharing
````
# Training
Download the HumanML3D dataset following the process here https://github.com/EricGuo5513/HumanML3D
Place the HumanML3D dataset in the MotionDataset/ directory and run the following python code.
````
python process_data.py
````
In order to train run the following command. Depending on the GPU used you may need to modify the config file. Our model was trained on a single 24GB NVIDIA RTX A5000 in approximatly 24 hours. Note our code has a known bug where it can only be ran on GPU 0 the first time its ran. After this first run the relevant files will be cahced and the training can be done on any GPU index. 
````
python train_primary_evaluator.py --device=0 --config=configs/base_config.yml
````
# Evaluation
To evaluate the model modify the file optimize_evalutator.py for the desired hyperparameter search options and run the following.
````
python optimize_evaluator.py --device=0 --checkpoint=../checkpoints/primary_evaluator/std_bpe2000/best_Faithfulness_checkpoint.pth
````
# Use As A Evaluator
To utilize as a evalutor follow these instructions.
````python
# Create instance of model, with path to tokenizer and regressor checkpoints
model = MotionTextEvalBERT(
    primary_evaluator_model_config, 
    chunk_encoder_config, 
    tokenizer_and_embedders_config,
    tokenizer_path="../checkpoints/primary_evaluator/std_bpe2000/tokenizer.tk",
    load_trained_regressors_path="../checkpoints/primary_evaluator/std_bpe2000/"
    )
# Send model to device
model = model.to(device = device)
# Load model checkpoint weights
checkpoint = torch.load("../checkpoints/primary_evaluator/std_bpe2000/best_Faithfulness_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
# Load motion and text data
# Motions should be a series of frame chunks (each 14 frames long) with consecutive chunks having a overlap of 4. 
# Each frame should be represented in the 263 dimensional representation as developed for HumanML3D. 
# Reference primary_evaluator_dataset.py for examples
motions, motion_masks = ... 
texts = ...
# Alignment is the non-finetuned alignment probability prediction as used in training and has gradients. 
# faithfulness_rating and naturalness_rating are human guidance finetuned ratings using sklearn SVR regression models over the model features (higher correlation than alignment).
# All scores should range from [0, 1] with higher scores being better. Regression scores may occur outside this range as well. 
alignment, faithfulness_rating, naturalness_rating = model.rate_alignment_batch(texts, motions, motion_masks, device)
````
# Acknologments
This code is built with some code provided by https://github.com/EricGuo5513/HumanML3D and we are greatful for the work they have done up to this point. 
