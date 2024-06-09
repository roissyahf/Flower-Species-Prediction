# Predict Flower Species with Pre-trained Model

This project will utilized pre-trained model: vgg11, vgg13, and vgg16 to classify different species of flowers. The model was trained on dataset comprises of 102 flowers category, in RGB color. **The first step** to do is experimentation, I used different pre-trained model configuration, until I obtained the best model. Loss in Training-Validation-Test set are: 2.183, 0.824, 0.9112 with Accuracy in Validation-Test set are: 0.780 and 0.751.

[Link to download the dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)

_Training Validation Acc-Loss_
![9f56c7b8e2f5d2928c45d6327b329a866286783d](https://github.com/roissyahf/Flower-Species-Prediction/assets/94748266/af324042-f013-4649-8460-a48b45b222a9)

_Example of prediction result_
![bd62740f4e4b1442df78e1b2610daad89a949587](https://github.com/roissyahf/Flower-Species-Prediction/assets/94748266/85592021-0e51-431a-b0a3-0de480df4b9c)

**The second step** I did, was code refactoring. Wrote simplified code version in Python script, so we can train the model and do prediction through CLI.
* `train.py` will train a new network on a dataset and save the model as a checkpoint. It will print out training loss, validation loss, and validation accuracy as the network trains.
* `predict.py` uses a trained network to predict the class for an input image. You'll to pass a single image `/path/to/image` and return the flower name and class probability.

## Set up environment
You can run it by following below step:
**1. Clone this repository**
`git clone https://github.com/roissyahf/Flower-Species-Prediction.git`
**2. Install required packages**
`pip install requirements.txt`
**3. Run `train.py` in the CLI to train the model**
`python train.py data_directory`

Options available are:
* Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
* Choose architecture: `python train.py data_dir --arch "vgg13"`
* Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
* Use GPU for training: `python train.py data_dir --gpu`

**4. Run python `predict.py` in the CLI to get prediction**
`python predict.py`

Options available are:
* Return top K most likely classes: `python predict.py input checkpoint --top_k 3`
* Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
* Use GPU for inference: `python predict.py input checkpoint --gpu`

