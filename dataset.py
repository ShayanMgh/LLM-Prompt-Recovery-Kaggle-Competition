# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

class PromptRecoveryDataset(Dataset):
    def __init__(self, original_texts, rewritten_texts, rewrite_prompts, tokenizer, max_length=512):
        self.original_texts = original_texts
        self.rewritten_texts = rewritten_texts
        self.rewrite_prompts = rewrite_prompts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract features
        self.features = self._extract_features()
    
    def _extract_features(self):
        """Extract features from text pairs"""
        features = []
        
        for orig, rewritten, prompt in zip(self.original_texts, self.rewritten_texts, self.rewrite_prompts):
            # Text length features
            orig_len = len(orig.split())
            rewritten_len = len(rewritten.split())
            len_diff = rewritten_len - orig_len
            len_ratio = rewritten_len / max(orig_len, 1)
            
            # Lexical features
            common_words = set(orig.lower().split()) & set(rewritten.lower().split())
            word_overlap = len(common_words) / max(len(set(orig.lower().split())), 1)
            
            # Style features
            orig_formal_score = self._formality_score(orig)
            rewritten_formal_score = self._formality_score(rewritten)
            formal_diff = rewritten_formal_score - orig_formal_score
            
            # Sentiment features
            orig_sentiment = self._sentiment_score(orig)
            rewritten_sentiment = self._sentiment_score(rewritten)
            sentiment_diff = rewritten_sentiment - orig_sentiment
            
            features.append({
                'len_diff': len_diff,
                'len_ratio': len_ratio,
                'word_overlap': word_overlap,
                'formal_diff': formal_diff,
                'sentiment_diff': sentiment_diff
            })
        
        return features
    
    def _formality_score(self, text):
        """Simple formality score based on word choices"""
        formal_words = ['therefore', 'furthermore', 'consequently', 'hence', 'thus', 'regarding']
        informal_words = ['like', 'so', 'pretty', 'kind of', 'sort of', 'basically']
        
        text_lower = text.lower()
        formal_count = sum(1 for word in formal_words if word in text_lower)
        informal_count = sum(1 for word in informal_words if word in text_lower)
        
        # Check for contractions
        contractions = len(re.findall(r"'s|'re|'ve|'ll|'m|'d|n't", text_lower))
        
        # Return score between -1 (informal) and 1 (formal)
        return (formal_count - informal_count - contractions) / (formal_count + informal_count + contractions + 1)
    
    def _sentiment_score(self, text):
        """Simple sentiment score based on word choices"""
        positive_words = ['good', 'great', 'excellent', 'best', 'happy', 'positive', 'wonderful', 'amazing']
        negative_words = ['bad', 'worst', 'terrible', 'awful', 'sad', 'negative', 'horrible', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Return score between -1 (negative) and 1 (positive)
        return (positive_count - negative_count) / (positive_count + negative_count + 1)
    
    def __len__(self):
        return len(self.original_texts)
    
    def __getitem__(self, idx):
        # Prepare input text
        input_text = f"Original: {self.original_texts[idx]}\nRewritten: {self.rewritten_texts[idx]}\nWhat was the rewrite instruction?"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize target
        labels = self.tokenizer(
            self.rewrite_prompts[idx],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.input_ids.squeeze(0)
        
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Add features
        feature_dict = self.features[idx]
        feature_tensor = torch.tensor([
            feature_dict['len_diff'],
            feature_dict['len_ratio'],
            feature_dict['word_overlap'],
            feature_dict['formal_diff'],
            feature_dict['sentiment_diff']
        ], dtype=torch.float)
        
        # Return all data
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'labels': labels,
            'features': feature_tensor
        }

# Example usage
def create_dataset_example():
    # Sample data
    data = {
        'original_text': ["The quick brown fox jumps over the lazy dog.", 
                          "I love programming and solving complex problems."],
        'rewritten_text': ["The swift brown fox leaps over the indolent canine.", 
                           "I enjoy coding and tackling intricate challenges."],
        'rewrite_prompt': ["Make the text more formal", 
                           "Use more sophisticated vocabulary"]
    }
    
    df = pd.DataFrame(data)
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    
    # Create dataset
    dataset = PromptRecoveryDataset(
        df['original_text'].tolist(),
        df['rewritten_text'].tolist(),
        df['rewrite_prompt'].tolist(),
        tokenizer
    )
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Print sample
    for batch in dataloader:
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention mask shape:", batch['attention_mask'].shape)
        print("Labels shape:", batch['labels'].shape)
        print("Features shape:", batch['features'].shape)
        break
    
    return dataset

if __name__ == "__main__":
    # Test the dataset
    dataset = create_dataset_example()
    print(f"Created dataset with {len(dataset)} examples")
