from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import tensorflow as tf

# Load the fine-tuned model
save_directory = "./saved_models"
model_fine_tuned = TFDistilBertForSequenceClassification.from_pretrained(save_directory)

# Ask the user for input text
user_input_text = input("Enter the text you want to classify: ")

# Tokenize the user's input
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
test_input = tokenizer(user_input_text, truncation=True, padding=True, return_tensors="tf")

# Ensure the input is a plain tensor
input_ids = test_input['input_ids']
attention_mask = test_input['attention_mask']

# Make predictions using the fine-tuned model
output = model_fine_tuned.predict([input_ids, attention_mask])
predicted_label = tf.argmax(output.logits, axis=1).numpy()[0]

# Define the label mapping from the configuration file
label_mapping = {
    0: "Business",
    1: "Entertainment",
    2: "Politics",
    3: "Sport",
    4: "Tech"
}

# Map the predicted label to a human-readable label
predicted_category = label_mapping.get(predicted_label, "Unknown")

print("Thank you for the news!")
print("I have classified this news as:", predicted_category)
