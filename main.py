# First, define all necessary functions and classes

import pandas as pd
import numpy as np
import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm.notebook import tqdm
import gc
import random
import time
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors

# Data Generation Function
def generate_synthetic_dataset(
    num_samples=1000,
    base_model_name="google/flan-t5-small",
    output_file="synthetic_dataset.csv",
    batch_size=4
):
    """
    Generate a synthetic dataset of original texts, rewrite prompts, and rewritten texts.
    """
    try:
        # Load tokenizer and model - use smaller models for Colab
        print(f"Loading {base_model_name} model...")
        tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Define rewriting instructions
        rewrite_instructions = [
            "Make the text more formal",
            "Make the text more casual",
            "Make the text more concise",
            "Make the text more elaborate",
            "Make the text more positive",
            "Make the text more negative",
            "Simplify the text",
            "Use more sophisticated vocabulary",
            "Rephrase to be more persuasive",
            "Rewrite to be more objective",
            "Change the tone to be more friendly",
            "Change the tone to be more serious",
            "Rewrite to be more engaging",
            "Restructure for better clarity",
            "Rewrite to sound more authoritative",
            "Rewrite to be more diplomatic",
            "Change to active voice",
            "Change to passive voice"
        ]
        
        # Sample texts to rewrite - start with a diverse set
        seed_texts = [
            "The company announced its quarterly earnings yesterday, which exceeded market expectations.",
            "I think we should consider other options before making a final decision on this matter.",
            "The research team has made significant progress in developing a new treatment for the disease.",
            "She wasn't happy with how the meeting went and decided to voice her concerns.",
            "The customer requested a refund because the product did not meet their expectations.",
            "The new policy will be implemented next month after a period of training and adjustment.",
            "I really enjoyed the movie despite what critics said about it.",
            "The data suggests that the intervention had a positive impact on student performance.",
            "He couldn't understand why his proposal was rejected without explanation.",
            "The committee will review all applications before making recommendations."
        ]
        
        # Initialize variables to store results
        original_texts = []
        rewrite_prompts = []
        rewritten_texts = []
        
        print(f"Generating {num_samples} samples in batches of {batch_size}...")
        
        # Function to generate text with T5
        def generate_text(prompt, max_length=100):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Use more conservative generation parameters for Colab
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,  # Reduced beam search
                no_repeat_ngram_size=2,
                early_stopping=True
            )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Generate new seed texts if needed
        if len(seed_texts) < num_samples // 10:
            topics = [
                "technology", "education", "healthcare", "environment", 
                "business", "politics", "science", "arts", "sports", "travel"
            ]
            
            print("Generating additional seed texts...")
            for topic in topics:
                if len(seed_texts) >= num_samples // 5:
                    break
                    
                try:
                    new_text = generate_text(f"Generate a paragraph about {topic}")
                    seed_texts.append(new_text)
                    
                    # Add memory management
                    if len(seed_texts) % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error generating seed text: {e}")
                    continue
        
        # Process in batches
        for i in tqdm(range(0, num_samples, batch_size)):
            batch_size_actual = min(batch_size, num_samples - i)
            
            for j in range(batch_size_actual):
                try:
                    # Select a random seed text
                    original_text = random.choice(seed_texts)
                    
                    # Select a random rewriting instruction
                    rewrite_prompt = random.choice(rewrite_instructions)
                    
                    # Generate rewritten text
                    input_prompt = f"Original: {original_text}\nInstruction: {rewrite_prompt}\nRewritten:"
                    rewritten_text = generate_text(input_prompt)
                    
                    # Fix common issues with generation
                    if "Rewritten:" in rewritten_text:
                        rewritten_text = rewritten_text.split("Rewritten:")[1].strip()
                    
                    # Add to dataset
                    original_texts.append(original_text)
                    rewrite_prompts.append(rewrite_prompt)
                    rewritten_texts.append(rewritten_text)
                    
                except Exception as e:
                    print(f"Error in batch processing: {e}")
                    # Add a simple fallback mechanism
                    original_text = "This is a sample text."
                    rewrite_prompt = "Make the text more formal"
                    rewritten_text = "This is a sample text that has been formalized."
                    
                    original_texts.append(original_text)
                    rewrite_prompts.append(rewrite_prompt)
                    rewritten_texts.append(rewritten_text)
            
            # Memory management
            if (i + batch_size_actual) % (5 * batch_size) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(0.5)  # Give system time to release resources
        
        # Create DataFrame
        data = {
            'original_text': original_texts,
            'rewrite_prompt': rewrite_prompts,
            'rewritten_text': rewritten_texts
        }
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Dataset saved to {output_file}")
        
        # Add features for pattern matching
        df['len_ratio'] = df.apply(lambda row: len(row['rewritten_text']) / max(1, len(row['original_text'])), axis=1)
        
        # Measure formality (placeholder - a more sophisticated measure would be better)
        # This is a simple proxy using avg word length
        def avg_word_length(text):
            words = re.findall(r'\b\w+\b', text)
            if not words:
                return 0
            return sum(len(word) for word in words) / len(words)
        
        df['formal_diff'] = df.apply(
            lambda row: avg_word_length(row['rewritten_text']) - avg_word_length(row['original_text']), 
            axis=1
        )
        
        # Memory management before exit
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return df
        
    except Exception as e:
        print(f"An error occurred in dataset generation: {e}")
        # Return a minimal dataset as fallback
        fallback_df = pd.DataFrame({
            'original_text': ["This is a sample text."] * 5,
            'rewrite_prompt': ["Make the text more formal"] * 5,
            'rewritten_text': ["This is a formal sample text."] * 5,
            'len_ratio': [1.1] * 5,
            'formal_diff': [0.5] * 5
        })
        return fallback_df

# Define the model class
class PromptRecoveryModel(nn.Module):
    def __init__(self, t5_model_name, device='cpu'):
        super(PromptRecoveryModel, self).__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.device = device
        self.model.to(device)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
    
    def _prepare_inputs(self, original_text, rewritten_text):
        input_text = f"Original: {original_text} Rewritten: {rewritten_text}"
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return inputs
    
    def _prepare_outputs(self, prompt):
        outputs = self.tokenizer(
            prompt,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return outputs
    
    def train_model(self, train_loader, val_loader, epochs=3, learning_rate=5e-5):
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.forward(input_ids, attention_mask, labels)
                loss = outputs.loss
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.forward(input_ids, attention_mask, labels)
                    loss = outputs.loss
                    
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # We'll save the model later
        
        return best_val_loss
    
    def predict_prompt(self, original_text, rewritten_text):
        self.model.eval()
        
        inputs = self._prepare_inputs(original_text, rewritten_text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load(self, path):
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.model.to(self.device)

class PatternBasedPromptRecovery:
    def __init__(self):
        self.knn = None
        self.patterns = {
            "Make the text more formal": {
                "conditions": [
                    lambda r: r['formal_diff'] > 0.1,
                    lambda r: r['len_ratio'] < 1.5
                ]
            },
            "Make the text more casual": {
                "conditions": [
                    lambda r: r['formal_diff'] < -0.1,
                    lambda r: "!" in r['rewritten_text'] or "..." in r['rewritten_text']
                ]
            },
            "Make the text more concise": {
                "conditions": [
                    lambda r: r['len_ratio'] < 0.8
                ]
            },
            "Make the text more elaborate": {
                "conditions": [
                    lambda r: r['len_ratio'] > 1.3
                ]
            },
            "Simplify the text": {
                "conditions": [
                    lambda r: r['formal_diff'] < -0.05,
                    lambda r: r['len_ratio'] < 1.2
                ]
            }
        }
    
    def train(self, train_df):
        # Extract features for KNN
        X = train_df[['len_ratio', 'formal_diff']].values
        y = train_df['rewrite_prompt'].values
        
        # Train KNN model
        self.knn = NearestNeighbors(n_neighbors=3)
        self.knn.fit(X)
        
        # Create mapping from index to prompt
        self.idx_to_prompt = {i: prompt for i, prompt in enumerate(y)}
        
        return self
    
    def predict(self, features, row=None):
        # First try pattern-based approach
        for prompt, pattern in self.patterns.items():
            if row is not None and all(condition(row) for condition in pattern["conditions"]):
                return prompt
        
        # If no pattern matches, use KNN
        if self.knn is not None and isinstance(features, dict):
            features_arr = np.array([[features['len_ratio'], features['formal_diff']]])
            distances, indices = self.knn.kneighbors(features_arr)
            
            # Return most common prompt among neighbors
            return self.idx_to_prompt[indices[0][0]]
        
        # Default fallback
        return "Make the text more formal"

class PromptRecoveryDataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length=512, max_target_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        original_text = row['original_text']
        rewritten_text = row['rewritten_text']
        prompt = row['rewrite_prompt']
        
        input_text = f"Original: {original_text} Rewritten: {rewritten_text}"
        
        # Tokenize inputs
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                prompt,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        
        item = {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': targets.input_ids.squeeze()
        }
        
        return item

class EnsemblePromptRecovery:
    def __init__(self, t5_model_name="google/flan-t5-small", device='cpu'):
        self.neural_model = PromptRecoveryModel(t5_model_name, device)
        self.pattern_model = PatternBasedPromptRecovery()
        self.device = device
        self.tokenizer = self.neural_model.tokenizer
    
    def train(self, train_df, val_df, epochs=3, batch_size=8, learning_rate=5e-5):
        # Train the pattern-based model
        self.pattern_model.train(train_df)
        
        # Train the neural model
        train_dataset = PromptRecoveryDataset(train_df, self.tokenizer)
        val_dataset = PromptRecoveryDataset(val_df, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return self.neural_model.train_model(
            train_loader, 
            val_loader, 
            epochs=epochs, 
            learning_rate=learning_rate
        )
    
    def evaluate(self, test_df, batch_size=8):
        test_dataset = PromptRecoveryDataset(test_df, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        self.neural_model.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.neural_model(input_ids, attention_mask, labels)
                loss = outputs.loss
                
                total_loss += loss.item()
        
        return total_loss / len(test_loader)
    
    def predict(self, original_text, rewritten_text, features=None):
        # Get neural model prediction
        neural_prediction = self.neural_model.predict_prompt(original_text, rewritten_text)
        
        # Get pattern-based prediction
        if features:
            # Create a row-like object for pattern matching
            row = {
                'original_text': original_text,
                'rewritten_text': rewritten_text,
                'len_ratio': features['len_ratio'],
                'formal_diff': features['formal_diff']
            }
            pattern_prediction = self.pattern_model.predict(features, row)
        else:
            # Calculate features
            len_ratio = len(rewritten_text) / max(1, len(original_text))
            
            # Simple formality measure
            def avg_word_length(text):
                words = re.findall(r'\b\w+\b', text)
                if not words:
                    return 0
                return sum(len(word) for word in words) / len(words)
            
            formal_diff = avg_word_length(rewritten_text) - avg_word_length(original_text)
            
            features = {'len_ratio': len_ratio, 'formal_diff': formal_diff}
            row = {
                'original_text': original_text,
                'rewritten_text': rewritten_text,
                'len_ratio': len_ratio,
                'formal_diff': formal_diff
            }
            pattern_prediction = self.pattern_model.predict(features, row)
        
        # Simple ensemble logic - prioritize pattern model's prediction if available
        if pattern_prediction and pattern_prediction != "Make the text more formal":
            return pattern_prediction
        return neural_prediction
    
    def save_model(self, path):
        self.neural_model.save(path)
        
    def load_model(self, path):
        self.neural_model.load(path)

# Now the main execution code
# Configuration
NUM_SAMPLES = 200  # Start with fewer samples for testing
MODEL_NAME = "google/flan-t5-small"  # Use a smaller model for Colab
OUTPUT_DIR = "prompt_recovery_model"
DATASET_FILE = "synthetic_dataset.csv"

# Step 1: Generate synthetic dataset if not already exists
if not os.path.exists(DATASET_FILE):
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(
        num_samples=NUM_SAMPLES,
        base_model_name=MODEL_NAME,
        output_file=DATASET_FILE,
        batch_size=4  # Small batch size for Colab
    )
else:
    print(f"Loading existing dataset from {DATASET_FILE}...")
    df = pd.read_csv(DATASET_FILE)
    
print(f"Dataset size: {len(df)}")
print(df.head())

# Step 2: Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# Step 3: Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = EnsemblePromptRecovery(t5_model_name=MODEL_NAME, device=device)

# Step 4: Train model with smaller batch size for Colab
if len(train_df) > 0:
    print("Training model...")
    model.train(
        train_df=train_df,
        val_df=val_df,
        epochs=2,  # Use fewer epochs for testing
        batch_size=4,  # Smaller batch size for Colab
        learning_rate=5e-5
    )

# Step 5: Evaluate model
if len(test_df) > 0:
    print("Evaluating model...")
    test_loss = model.evaluate(test_df, batch_size=4)  # Smaller batch size
    print(f"Test Loss: {test_loss:.4f}")

# Step 6: Make some example predictions
if len(test_df) > 0:
    print("\nSample predictions:")
    for i in range(min(5, len(test_df))):
        sample = test_df.iloc[i]
        original_text = sample['original_text']
        rewritten_text = sample['rewritten_text']
        true_prompt = sample['rewrite_prompt']
        
        # Get features for pattern-based approach
        features = {
            'len_ratio': sample['len_ratio'],
            'formal_diff': sample['formal_diff']
        }
        
        predicted_prompt = model.predict(
            original_text=original_text,
            rewritten_text=rewritten_text,
            features=features
        )
        
        print(f"\nExample {i+1}:")
        print(f"Original: {original_text[:100]}...")
        print(f"Rewritten: {rewritten_text[:100]}...")
        print(f"True prompt: {true_prompt}")
        print(f"Predicted prompt: {predicted_prompt}")
        time.sleep(0.5)  # Prevent excessive output

# Step 7: Save model
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

print("\nDone!")
