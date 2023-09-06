# BBClyDistilBERT | News Classification with DistilBERT

This repository showcases the fine-tuning of Google's DistilBERT model for classifying news articles into one of the following categories such as tech, business, politics, entertainment and sports. Included are the preprocessing data(learning.py) to train the model, and prediction scripts(predict.py) for the fine-tuned model. Please note that due to size limitations, the fine-tuned model itself is not included.

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

(Include any results, accuracy metrics, or performance details achieved with your model here, if applicable.)

## Getting Started

To begin using this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/news-classification.git
Install the required dependencies:

pip install transformers tensorflow pandas scikit-learn

## Data Preprocessing
Before using the model, you should preprocess your dataset. If you choose to use your own custom dataset, follow these steps:

1. Tokenize your dataset and assign labels to each news article.

2. Convert labels to binary format, ensuring they are compatible with the model's requirements.

3. For a quick start, you can use the provided example dataset (BBC Text Classification) and follow the preprocessing steps outlined in the code.

4. Fine-tune the DistilBERT model on your preprocessed dataset.

5. Save the fine-tuned model in the saved_models directory. (Make sure you have permissions to write in the current pat of your OS)

## Usage
You can use the fine-tuned DistilBERT model to classify news articles. Run the following script and enter the news article text when prompted:

bash
Copy code
python predict.py
The script will classify the news article into one of the following categories: Business, Entertainment, Politics, Sport, Tech based on the given input.

## Model Fine-Tuning
If you want to fine-tune the DistilBERT model on your own dataset, follow these steps:

Prepare your dataset in a format similar to the example dataset (BBC Text Classification).

Modify the code to load and preprocess your dataset.

Fine-tune the model using the TFTrainer.

Save the fine-tuned model in the saved_models directory.

## Contributing
Contributions to this project are welcome! If you encounter issues or have suggestions for improvements, please open an issue or create a pull request.

## Additional Notes
Please be aware that training time may vary depending on the dataset size and the number of epochs used for training.
Currently working on train.py file

## Credits
@ github.com/rohan-paul
