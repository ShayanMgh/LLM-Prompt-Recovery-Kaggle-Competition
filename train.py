# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import difflib
import re
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# Load the data
train = pd.read_csv('llm-prompt-recovery/train.csv')
test = pd.read_csv('llm-prompt-recovery/test.csv')

# Display basic info
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("\nTrain columns:", train.columns.tolist())
print("\nTest columns:", test.columns.tolist())

# Display sample data
print("\nTrain sample:")
print(train.head(2))

def analyze_examples(df):
    """Analyze patterns in the examples"""
    
    print(f"Number of examples: {len(df)}")
    
    # Check if rewrite_prompt column exists
    if 'rewrite_prompt' not in df.columns:
        print("\nError: 'rewrite_prompt' column not found in the DataFrame")
        print("Available columns:", df.columns.tolist())
        return
    
    # Check if there are common patterns in the rewrite prompts
    prompt_counts = df['rewrite_prompt'].value_counts()
    print(f"\nNumber of unique rewrite prompts: {len(prompt_counts)}")
    print("\nTop 5 most common rewrite prompts:")
    print(prompt_counts.head(min(5, len(prompt_counts))))
    
    # Analyze prompt structure
    print("\nSample prompt analysis:")
    # Sample size will be the minimum of 5 and the total number of rows
    sample_size = min(5, len(df))
    
    for i, prompt in enumerate(df['rewrite_prompt'].sample(n=sample_size).tolist()):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Check if prompt starts with action verb
        first_word = prompt.split()[0].lower()
        print(f"First word: {first_word}")
        
        # Check if it has quotes
        has_quotes = '"""' in prompt or '"' in prompt
        print(f"Contains quotes: {has_quotes}")
        
        # Check length
        print(f"Length: {len(prompt)} characters, {len(prompt.split())} words")

# Let's also add some basic data validation before analysis
def validate_and_analyze_data(df):
    """Validate the data structure and perform analysis"""
    print("Data Validation:")
    print("-" * 50)
    print(f"Total rows: {len(df)}")
    print(f"Columns present: {df.columns.tolist()}")
    
    # Check for required columns
    required_columns = ['id', 'original_text', 'rewrite_prompt', 'rewritten_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print("\nWarning: Missing required columns:", missing_columns)
        print("Please ensure your data has the correct structure")
        return False
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("\nWarning: Found null values:")
        print(null_counts[null_counts > 0])
    
    print("\nBeginning Analysis:")
    print("-" * 50)
    analyze_examples(df)
    return True

try:
    # Load the data
    train = pd.read_csv('llm-prompt-recovery/train.csv')
    
    # Validate and analyze
    validate_and_analyze_data(train)
    
except Exception as e:
    print(f"Error occurred: {str(e)}")

def extract_features(row):
    """Extract features from original and rewritten text"""
    original = row['original_text']
    rewritten = row['rewritten_text']
    
    features = {}
    
    # Length features
    features['original_length'] = len(original)
    features['rewritten_length'] = len(rewritten)
    features['length_ratio'] = features['rewritten_length'] / max(1, features['original_length'])
    
    # Style features
    features['original_caps_ratio'] = sum(1 for c in original if c.isupper()) / max(1, len(original))
    features['rewritten_caps_ratio'] = sum(1 for c in rewritten if c.isupper()) / max(1, len(rewritten))
    
    # Check for specific transformations
    features['has_verse'] = 'verse' in rewritten.lower()
    features['has_chorus'] = 'chorus' in rewritten.lower()
    features['has_rhyme'] = False  # Would need more complex analysis
    features['is_bullet_points'] = rewritten.count('•') > 0 or rewritten.count('-') > 3
    features['is_numbered_list'] = bool(re.search(r'\d+\.\s', rewritten))
    
    # Vocabulary shift
    common_words = set(original.lower().split()).intersection(set(rewritten.lower().split()))
    features['common_words_ratio'] = len(common_words) / max(1, len(set(original.lower().split())))
    
    return pd.Series(features)

# Extract features
train_features = train.apply(extract_features, axis=1)

# Display sample features
print(train_features.head())

class PromptRecoveryDataset(Dataset):
    def __init__(self, df, tokenizer=None, max_length=512, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Format input as concatenation of original and rewritten text
        input_text = f"Original: {row['original_text']} Rewritten: {row['rewritten_text']}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
        }
        
        # For training data, also tokenize prompts
        if self.is_train:
            target_text = row['rewrite_prompt']
            targets = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            item['labels'] = targets['input_ids'].squeeze()
            
        return item

def train_prompt_recovery_model(train_df, val_df=None):
    # Initialize tokenizer and model
    model_name = "t5-large"  # Larger model for better performance
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Create datasets
    train_dataset = PromptRecoveryDataset(train_df, tokenizer, is_train=True)
    
    # Validation dataset if provided
    val_dataset = None
    if val_df is not None:
        val_dataset = PromptRecoveryDataset(val_df, tokenizer, is_train=True)
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./prompt_recovery_results",
        evaluation_strategy="epoch" if val_dataset else "no",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=True,  # Use mixed precision if you have a GPU
        gradient_accumulation_steps=4,
        load_best_model_at_end=True if val_dataset else False,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save model and tokenizer
    model.save_pretrained("./prompt_recovery_model")
    tokenizer.save_pretrained("./prompt_recovery_tokenizer")
    
    return model, tokenizer

# Split data for training and validation
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

# Train model
model, tokenizer = train_prompt_recovery_model(train_df, val_df)





