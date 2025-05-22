# model.py

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from huggingface_hub import HfFolder, Repository, push_to_hub

class EnsemblePromptRecovery:
    def __init__(self, model_name="google/flan-t5-large"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.pattern_matcher = PatternBasedPromptRecovery()
        
    def train(self, train_data, val_data=None, epochs=5, batch_size=8, learning_rate=5e-5):
        """Train the T5 model on the synthetic data"""
        if val_data is None:
            train_data, val_data = train_test_split(train_data, test_size=0.1)
        
        # Create datasets
        train_dataset = PromptRecoveryDataset(
            train_data['original_text'].tolist(),
            train_data['rewritten_text'].tolist(),
            train_data['rewrite_prompt'].tolist(),
            self.tokenizer
        )
        
        val_dataset = PromptRecoveryDataset(
            val_data['original_text'].tolist(),
            val_data['rewritten_text'].tolist(),
            val_data['rewrite_prompt'].tolist(),
            self.tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            # Print metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save the model locally
        self.model.save_pretrained("./prompt_recovery_model")
        self.tokenizer.save_pretrained("./prompt_recovery_model")
        
        # Push model to Hugging Face Hub
        self.push_to_hub("prompt_recovery_model")
    
    def push_to_hub(self, repo_name, private=False):
        """Push the trained model to Hugging Face Hub"""
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must login to the Hugging Face Hub first. Use `huggingface-cli login` or `notebook_login()`")
        
        repo = Repository("./prompt_recovery_model", clone_from=repo_name, use_auth_token=token)
        
        commit_message = "Add trained prompt recovery model"
        repo.push_to_hub(commit_message=commit_message, private=private)
        
        print(f"Model pushed to https://huggingface.co/{repo_name}")
    
    def predict(self, original_text, rewritten_text, ensemble_weight=0.7):
        """Predict rewrite prompt using both T5 model and pattern matching"""
        # T5 model prediction
        input_text = f"Original: {original_text}\nRewritten: {rewritten_text}\nWhat was the rewrite instruction?"
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        model_prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Pattern-based prediction
        pattern_prediction = self.pattern_matcher.predict(original_text, rewritten_text)
        
        # Combine predictions
        if ensemble_weight >= 0.9:
            return model_prediction
        elif ensemble_weight <= 0.1:
            return pattern_prediction
        else:
            # Check confidence of model prediction
            if len(model_prediction.split()) > 3 and any(keyword in model_prediction.lower() for keyword in ["convert", "rewrite", "transform"]):
                return model_prediction
            else:
                return pattern_prediction
