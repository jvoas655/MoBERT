# Acknologments
This code is built on top of code provided by https://github.com/EricGuo5513/HumanML3D and we are greatful for the work they have done up to this point. 
# MoBERT
MoBERT: A Strongly Correlated Text-to-Motion Evaluation Metric
# Installation
This code is tested on Linux Ubuntu 22.04 and with python 3.9. 
Using Conda run the following to create a enviroment.
````
conda env create -f environment.yml
````
Download checkpoints from this drive and unzip them in the checkpoints/primary_evaluator/ directory
````
https://drive.google.com/file/d/1dIokQMXqDSQZItk5Lt9CcQ2BTbhuTSRI/view?usp=sharing
````
Download our evaluation dataset from drive and unzip them in the MotionDataset/ folder.
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
````
TODO
````
