# Fine-Tuning DistilBERT for News Classification


This project demonstrates the fine-tuning of Google's DistilBERT model for classifying news articles into different categories. It includes both the training code and a simple interactive script to classify new text.
Please not that I was not able to upload the fine-tuned model because of the size limitations. I uploaded files for preprocessing data, training the model and using the finely tuned model for predictions
## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Fine-Tuning](#model-fine-tuning)
- [Contributing](#contributing)
- [License](#license)

## Introduction

News Classification with DistilBERT is a project that uses state-of-the-art natural language processing (NLP) techniques to classify news articles into predefined categories. This project utilizes [Hugging Face Transformers](https://huggingface.co/transformers/) and [TensorFlow](https://www.tensorflow.org/).

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/news-classification.git
Install the required dependencies:

bash
Copy code
pip install transformers tensorflow pandas scikit-learn
Fine-tune the DistilBERT model on your own dataset or use the provided example dataset (BBC Text Classification).

Save the fine-tuned model in the saved_models directory.

Usage
You can use the fine-tuned DistilBERT model to classify news articles. Run the following script and enter the news article text when prompted:

bash
Copy code
python predict.py
The script will classify the news article into one of the following categories: Business, Entertainment, Politics, Sport, Tech based on the give input

Model Fine-Tuning
If you want to fine-tune the DistilBERT model on your own dataset, follow these steps:

Prepare your dataset in a format similar to the example dataset (BBC Text Classification).

Modify the code to load and preprocess your dataset.

Fine-tune the model using the TFTrainer.

Save the fine-tuned model in the saved_models directory.

Contributing
Contributions to this project are welcome! If you find issues or have suggestions for improvements, please open an issue or create a pull request.

