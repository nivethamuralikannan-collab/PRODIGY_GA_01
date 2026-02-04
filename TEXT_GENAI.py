# ============================================
# GPT-2 TEXT GENERATION (BEGINNER FRIENDLY)
# ============================================

# Step 1: Install required libraries (run once)
# Uncomment the next line if running locally
!pip install transformers torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Step 2: Load pretrained GPT-2 tokenizer and model
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Set the model to evaluation mode
model.eval()

# Step 4: Enter your input prompt here
prompt = "ice cream"

# Step 5: Convert input text to tokens
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Step 6: Generate text using GPT-2
output_ids = model.generate(
    input_ids=input_ids,
    max_length=80,        # Maximum length of generated text
    temperature=0.7,      # Controls creativity (lower = safer)
    top_k=50,             # Limits sampling to top-k tokens
    top_p=0.95,           # Nucleus sampling
    do_sample=True,       # Enables randomness
    num_return_sequences=1
)

# Step 7: Decode generated tokens back to text
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Step 8: Print the result
print("\nGenerated Text:\n")
print(generated_text)