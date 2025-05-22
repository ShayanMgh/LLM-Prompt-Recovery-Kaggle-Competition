# data_generation.py

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import random
import pandas as pd
from tqdm import tqdm
import gc
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Source texts for generation
source_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "I love programming and solving complex problems.",
    "Machine learning is transforming the world.",
    "The sun sets in the west and rises in the east.",
    "Climate change is a global challenge we must address.",
    # Add more source texts as needed
]

# Prompt templates
PROMPT_TEMPLATES = [
    "Make the text more formal",
    "Convert to a casual tone",
    "Rewrite in a professional style",
    "Make it more concise",
    "Elaborate and add more details",
    "Change to passive voice",
    "Change to active voice",
    "Make it more positive",
    "Make it more negative",
    "Simplify the language",
]

# Pattern-based transformations
PATTERNS = {
    "formal": {
        "words": ["therefore", "furthermore", "consequently", "hence"],
        "structures": ["It is evident that", "One must consider", "It can be concluded that"]
    },
    "casual": {
        "words": ["basically", "pretty much", "kind of", "like"],
        "structures": ["You know what?", "Check this out:", "Here's the thing:"]
    },
    "professional": {
        "words": ["implement", "facilitate", "optimize", "leverage"],
        "structures": ["Based on the analysis", "From a professional standpoint", "In consideration of"]
    }
}

def generate_with_patterns(n_samples):
    """Generate examples using pattern-based approach"""
    examples = []
    for _ in range(n_samples):
        # Select random source text
        original_text = random.choice(source_texts)
        
        # Select random transformation type
        transform_type = random.choice(list(PATTERNS.keys()))
        pattern = PATTERNS[transform_type]
        
        # Apply transformation
        rewritten_text = original_text
        if random.random() < 0.7:  # 70% chance to add introductory phrase
            rewritten_text = random.choice(pattern["structures"]) + " " + rewritten_text
        
        # Add style-specific words
        words = pattern["words"]
        if words and random.random() < 0.8:  # 80% chance to add style-specific word
            insert_pos = random.randint(0, len(rewritten_text.split()))
            rewritten_words = rewritten_text.split()
            rewritten_words.insert(insert_pos, random.choice(words))
            rewritten_text = " ".join(rewritten_words)
        
        # Generate rewrite prompt
        rewrite_prompt = f"Transform the text to be more {transform_type}"
        
        examples.append({
            "original_text": original_text,
            "rewritten_text": rewritten_text,
            "rewrite_prompt": rewrite_prompt
        })
    
    return examples

def generate_with_flant5(n_samples):
    """Generate examples using Flan-T5"""
    # Use smaller model to prevent memory issues
    model_name = "google/flan-t5-base"
    
    try:
        # Load model and tokenizer with memory optimization
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(device)
        
        examples = []
        for _ in range(n_samples):
            try:
                # Get source text
                original_text = random.choice(source_texts)
                
                # Generate rewrite instruction
                prompt_template = random.choice(PROMPT_TEMPLATES)
                input_text = f"Generate a rewrite instruction: {prompt_template}"
                
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.92,
                        top_k=50,
                        early_stopping=True
                    )
                rewrite_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Generate rewritten text
                input_text = f"Original: {original_text}\nInstruction: {rewrite_prompt}\nRewritten:"
                inputs = tokenizer(input_text, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        num_beams=1,
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.92,
                        top_k=50,
                        early_stopping=True
                    )
                rewritten_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                examples.append({
                    "original_text": original_text,
                    "rewritten_text": rewritten_text,
                    "rewrite_prompt": rewrite_prompt
                })
                
            except Exception as e:
                print(f"Error generating single example: {e}")
                continue
                
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        return examples
        
    except Exception as e:
        print(f"Error initializing Flan-T5: {e}")
        return []

def generate_synthetic_dataset(n_samples=1000, model_names=["pattern"], batch_size=4):
    """Generate synthetic dataset using both T5 and pattern-based approaches"""
    print(f"Device set to use {device}")
    
    synthetic_data = []
    total_generated = 0
    
    with tqdm(total=n_samples) as pbar:
        while total_generated < n_samples:
            # Memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            current_batch_size = min(batch_size, n_samples - total_generated)
            if current_batch_size <= 0:
                break
            
            # Select generation method
            model_name = random.choice(model_names)
            
            try:
                if model_name == "flant5":
                    batch_data = generate_with_flant5(current_batch_size)
                    if not batch_data:  # If T5 generation failed
                        batch_data = generate_with_patterns(current_batch_size)
                else:
                    batch_data = generate_with_patterns(current_batch_size)
                
                synthetic_data.extend(batch_data)
                total_generated += current_batch_size
                pbar.update(current_batch_size)
                
                print(f"Generated {total_generated}/{n_samples} examples")
                
            except Exception as e:
                print(f"Error in batch generation: {e}")
                continue
    
    return pd.DataFrame(synthetic_data)

def main():
    # Generate dataset with only pattern-based generation first
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=100, model_names=["pattern"])
    
    # Save the dataset
    df.to_csv("synthetic_data.csv", index=False)
    print(f"Generated {len(df)} examples and saved to synthetic_data.csv")
    
    # Print a sample
    print("\nSample of generated data:")
    print(df.head())

if __name__ == "__main__":
    main()
