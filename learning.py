from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
import tensorflow as tf
import pandas as pd
import os

from sklearn.model_selection import train_test_split

root_path = 'bbc-text.csv'
df = pd.read_csv(root_path)

#Found a bug
#df['category'].unique()

#Fixed to:
unique_categories = df['category'].unique()

# Ensure 'category' is unique
if len(unique_categories) != len(df['category']):
    # Handle non-unique values by dropping duplicates
    df.drop_duplicates(subset=['category'], inplace=True)

#print(df['category'])

#'category' column is being converted into numerical labels
df['encoded_text'] = df['category'].astype('category').cat.codes

data_texts = df['text'].to_list()
data_labels = df['encoded_text'].to_list()

# print(data_labels)
# print(data_texts)


#Train Test Split
train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size = 0.2, random_state = 0 )
train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size = 0.01, random_state = 0 )

#Model Definition
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation = True, padding = True  )
val_encodings = tokenizer(val_texts, truncation = True, padding = True )

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
))
 
#Fine-tuning with the TFTrainer class
model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

training_args = TFTrainingArguments(
    output_dir='./results',          
    num_train_epochs=7,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=64,   
    warmup_steps=500,                
    weight_decay=1e-5,               
    logging_dir='./logs',            
    eval_steps=100                   
)

# with training_args.strategy.scope():
trainer_model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels = 5 )


trainer = TFTrainer(
    model=trainer_model,                 
    args=training_args,                  
    train_dataset=train_dataset,         
    eval_dataset=val_dataset,            
)

trainer.train()
trainer.evaluate()

save_directory = "./saved_models" 

# Ensure the directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

model.save_pretrained(save_directory)

