import pandas as pd
import sys
import io
import torch  
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from transformers import BertModel, BertTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') 

# Read user features from raw data file
user_original_df = pd.read_csv(r"datafiles\\COPE\\merged_user_files_1.csv", 
                 encoding='utf-8', 
                 on_bad_lines='skip',
                 # nrows=200 
                 )

# Read predicted text file
user_predication_df = pd.read_csv(r"outputFiles\\CoT_rolePlaying_1.csv", 
                 encoding='utf-8', 
                 on_bad_lines='skip',
                 )

predicate_texts = user_predication_df['predicative_text'].tolist()  # Extract predicted text data  
## Use pre-trained panic emotion recognition model to classify generated text
# 1. Define model class  
class PanicClassificationModel(nn.Module):  
    def __init__(self, bert_name):  
        super(PanicClassificationModel, self).__init__()  
        self.bert = BertModel.from_pretrained(bert_name)  
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classification  

    def forward(self, input_ids, attention_mask):  
        outputs = self.bert(input_ids, attention_mask=attention_mask)  
        sequence_output = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token output  
        logits = self.classifier(sequence_output)  
        return logits  

# 2. Load saved model  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
model_path = r'modelFiles\\panic_model.pth'   
model = PanicClassificationModel(bert_name='bert-base-uncased').to(device)  
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()  

# 3. Perform text classification (skip NaN and preserve empty values)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  
max_length = 50  
# Define classify_texts function to process and classify text  
def classify_texts(texts):
    predictions = []
    with torch.no_grad():
        for text in texts:
            if pd.isna(text) or not isinstance(text, str):
                predictions.append(None) 
                continue  
            
            encoding = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            predictions.append(preds.item())
    
    # Convert None to NaN for empty storage in DataFrame
    return [np.nan if pred is None else pred for pred in predictions]
 
predictions = classify_texts(predicate_texts)   

# 4. Add prediction results to original DataFrame
user_predication_df['panic_related'] = predictions  
user_predication_df.to_csv(
        r"outputFiles\\CoT_rolePlaying_paniceRelated.csv",
        index=False,
        encoding='utf-8-sig'  
    )

# 6. Predict panic users
# Create panic_user field, check if each user_id has any post with panic_related = 1  
panic_users = user_predication_df.groupby('user_id')['panic_related'].max().reset_index()  
panic_users.columns = ['user_id', 'predicative_panic_user']  
panic_users['predicative_panic_user'] = (panic_users['predicative_panic_user'] == 1).astype(int)  

# 7. Calculate and print prediction results
# Extract true values from user_df
user_df = user_original_df[['user_id', 'panic_user']] 
merged_df = pd.merge(user_df, panic_users, on='user_id', how='inner') 

# Extract true labels and predicted labels from merged DataFrame  
true_labels = merged_df['panic_user'].tolist()  
predicted_labels = merged_df['predicative_panic_user'].astype(int).tolist()  

# Print classification report and other metrics
print("Accuracy:", accuracy_score(true_labels, predicted_labels))
print("AUC:", roc_auc_score(true_labels, predicted_labels))
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=['No Panic', 'Panic'])) 