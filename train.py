from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf

# Load the fine-tuned model
save_directory = "./saved_models"
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

# Define the test text
test_text = 'dollar hovers around record lows the us dollar hovered close to record lows against the euro on friday...'

# Tokenize the test text (you can omit this part if already tokenized)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
test_input = tokenizer(test_text, truncation=True, padding=True, return_tensors="tf")

# Make predictions using the fine-tuned model
output = model_fine_tuned.predict(test_input)
prediction_value = tf.argmax(output.logits, axis=1).numpy()[0]

print("Predicted Label:", prediction_value)
