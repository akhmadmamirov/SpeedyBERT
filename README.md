# SpeedyBERT | News Classification with DistilBERT

This repository showcases the fine-tuning of Google's DistilBERT model for classifying news articles into one of the following categories such as tech, business, politics, entertainment and sports. Included are the preprocessing data(learning.py) to train the model, and prediction scripts(predict.py) for the fine-tuned model. Please note that due to size limitations, the fine-tuned model itself is not included.

<img width="581" alt="image" src="https://github.com/akhmadmamirov/fineTuningBert/assets/105142060/e6357cd2-3d88-465c-a0cd-a2117765fe3f">


## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Usage](#usage)
- [Model Fine-Tuning](#model-fine-tuning)
- [Contributing](#contributing)

## Introduction

News Classification with DistilBERT is a project that harnesses cutting-edge natural language processing (NLP) techniques to categorize news articles into predefined topics. This project relies on [Hugging Face Transformers](https://huggingface.co/transformers/) and [TensorFlow](https://www.tensorflow.org/) for its implementation.

## Achieved Results

1. Built a NLP Pipeline to train Google’s DistilBert Large Language Model using TensorFlow and Hugging Face transformers for multi class text classification.
2. Fine-Tuned the model with a custom BBC text classification dataset.
3. Used DistilBert to pertain 97% of language understanding of Bert Model while reducing the size by 40% and speeding up the training process by 60%.



## Getting Started

To begin using this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/akhmadmamirov/fineTuningBert.git
Install the required dependencies:

pip install transformers tensorflow pandas scikit-learn

## Data Preprocessing
Before using the model, you should preprocess your dataset. If you choose to use your own custom dataset, follow these steps:

1. Tokenize your dataset and assign labels to each news article.

2. Convert labels to binary format, ensuring they are compatible with the model's requirements.

3. For a quick start, you can use the provided example dataset (BBC Text Classification) and follow the preprocessing steps outlined in the code.

4. Fine-tune the DistilBERT model on your preprocessed dataset.

5. Save the fine-tuned model in the saved_models directory. (Make sure you have permissions to write in the current path of your OS)

## Usage
You can use the fine-tuned DistilBERT model to classify news articles. Run the following script and enter the news article text when prompted:

bash
Copy code
python predict.py
The script will classify the news article into one of the following categories: Business, Entertainment, Politics, Sport, Tech based on the given input.

## Model Fine-Tuning
If you want to fine-tune the DistilBERT model on your own dataset, follow these steps:

1. Prepare your dataset in a format similar to the example dataset (BBC Text Classification).

2. Modify the code to load and preprocess your dataset.

3. Fine-tune the model using the TFTrainer.

4. Save the fine-tuned model in the saved_models directory.

## Contributing
Contributions to this project are welcome! If you encounter issues or have suggestions for improvements, please open an issue or create a pull request.

## Additional Notes
1. Please be aware that training time may vary depending on the dataset size and the number of epochs used for training.
2. The average training time in my case was <7 hours
3. Currently working on train.py file

## Credits
1. @ github.com/rohan-paul
2. ChatGPT 
