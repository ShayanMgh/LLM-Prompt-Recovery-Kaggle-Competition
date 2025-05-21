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

