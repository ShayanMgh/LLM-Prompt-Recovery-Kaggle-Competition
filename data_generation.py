import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import random
from tqdm import tqdm

def generate_synthetic_dataset(n_samples=1000):
    """Generate synthetic training data using Gemma 7b-it"""
    
    # Load Gemma model (or a similar model available on Colab)
    model_name = "google/gemma-7b-it"  # Use this if you have access
    # Alternative: "google/flan-t5-xl" or "google/flan-ul2" which are available on Colab
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    except:
        print("Falling back to T5 model")
        from transformers import T5ForConditionalGeneration
        model_name = "google/flan-t5-xl"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0)
    
    # Create a diverse set of original texts
    original_texts = []
    
    # Academic/formal texts
    academic_prompts = [
        "Write a paragraph about climate change impacts",
        "Explain quantum computing in simple terms",
        "Describe the process of photosynthesis",
        "Discuss the implications of artificial intelligence on society"
    ]
    
    # News articles
    news_prompts = [
        "Write a news article about a recent technological breakthrough",
        "Create a news report about an economic development",
        "Write a short news piece about a scientific discovery"
    ]
    
    # Creative writing
    creative_prompts = [
        "Write a short story about a mysterious island",
        "Create a descriptive paragraph about a forest",
        "Write about a character who discovers a hidden ability"
    ]
    
    # Generate original texts
    all_prompts = academic_prompts + news_prompts + creative_prompts
    for _ in tqdm(range(n_samples // len(all_prompts) + 1)):
        for prompt in all_prompts:
            if len(original_texts) >= n_samples:
                break
                
            try:
                if "flan-t5" in model_name:
                    result = generator(prompt, max_length=200)[0]["generated_text"]
                else:
                    result = generator(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
                    if prompt in result:  # Remove the prompt from the output
                        result = result[len(prompt):].strip()
                
                original_texts.append(result)
            except Exception as e:
                print(f"Error generating text: {e}")
    
    # Create rewrite prompts (diverse transformation instructions)
    rewrite_templates = [
        # Style transformations
        "Convert this into a sea shanty: \"\"\"{}\"\"\"",
        "Rewrite this as a Shakespearean sonnet: \"\"\"{}\"\"\"",
        "Transform this into a pirate's speech: \"\"\"{}\"\"\"",
        "Rewrite this in the style of a legal document: \"\"\"{}\"\"\"",
        "Rewrite this as a series of haikus: \"\"\"{}\"\"\"",
        "Convert this into a rap song: \"\"\"{}\"\"\"",
        "Rewrite this in the style of Ernest Hemingway: \"\"\"{}\"\"\"",
        "Transform this into a fairy tale: \"\"\"{}\"\"\"",
        
        # Tone transformations
        "Make this text more formal: \"\"\"{}\"\"\"",
        "Make this text more casual: \"\"\"{}\"\"\"",
        "Rewrite this to be more enthusiastic: \"\"\"{}\"\"\"",
        "Make this text sound more academic: \"\"\"{}\"\"\"",
        "Rewrite this to be more concise: \"\"\"{}\"\"\"",
        "Make this text more dramatic: \"\"\"{}\"\"\"",
        
        # Format transformations
        "Convert this into a dialogue between two people: \"\"\"{}\"\"\"",
        "Rewrite this as a bullet point list: \"\"\"{}\"\"\"",
        "Transform this into a step-by-step guide: \"\"\"{}\"\"\"",
        "Rewrite this as a FAQ section: \"\"\"{}\"\"\"",
        "Convert this into a news headline and brief: \"\"\"{}\"\"\"",
        
        # Content transformations
        "Simplify this text for a 5th-grade reading level: \"\"\"{}\"\"\"",
        "Explain this concept as if I'm five years old: \"\"\"{}\"\"\"",
        "Add more technical details to this text: \"\"\"{}\"\"\"",
        "Rewrite this to include more examples: \"\"\"{}\"\"\"",
        "Transform this into a persuasive argument: \"\"\"{}\"\"\"",
        "Rewrite this to be more politically neutral: \"\"\"{}\"\"\"",
        "Add humor to this text: \"\"\"{}\"\"\"",
        "Make this text more poetic: \"\"\"{}\"\"\"",
    ]
    
    # Generate dataset
    data = []
    for i, original_text in enumerate(original_texts[:n_samples]):
        rewrite_prompt = random.choice(rewrite_templates).format(original_text)
        
        try:
            # Generate rewritten text using the model
            if "flan-t5" in model_name:
                rewritten_text = generator(rewrite_prompt, max_length=300)[0]["generated_text"]
            else:
                rewritten_text = generator(rewrite_prompt, max_new_tokens=300, do_sample=True)[0]["generated_text"]
                if rewrite_prompt in rewritten_text:  # Remove the prompt from the output
                    rewritten_text = rewritten_text[len(rewrite_prompt):].strip()
            
            data.append({
                'id': i,
                'original_text': original_text,
                'rewrite_prompt': rewrite_prompt,
                'rewritten_text': rewritten_text
            })
        except Exception as e:
            print(f"Error generating rewritten text: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.to_csv('synthetic_training_data.csv', index=False)
    print(f"Generated {len(df)} synthetic training examples")
    return df

# Generate synthetic data
synthetic_data = generate_synthetic_dataset(n_samples=500)  # Start with 500 examples
