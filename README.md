# GPT-2 Text Generation (Beginner Friendly)
## What is this project?
This project demonstrates text generation using GPT-2, a pretrained language model from OpenAI, implemented using the Hugging Face Transformers library in Python.
The model takes a prompt (example: "ice cream") and generates meaningful text based on it.
---
## 🎯 Why use GPT-2?
GPT-2 is a pretrained language model
It can generate human-like text
No training required — works out of the box
Ideal for beginners in NLP and AI
---
## 🛠️ Technologies Used
Python
PyTorch
Hugging Face Transformers
GPT-2 Model
---
## 📂 Project Structure
```
gpt2-text-generation/
│
├── gpt2_text_generation.py
└── README.md
```
## ⚙️ Installation
### 1️⃣ Install Python packages
Make sure Python 3.7+ is installed.
```

pip install torch transformers

```
### ▶️ How to Run the Program
```
python gpt2_text_generation.py
```
### 🧾 Code Explanation (What each step does)
#### Step 1: Load model and tokenizer

Python
```
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
```
➡ Loads pretrained GPT-2 model and tokenizer.
#### Step 2: Set evaluation mode

Python
```
model.eval()
```
➡ Disables training behavior and enables inference.
#### Step 3: Provide input prompt

Python
```
prompt = "ice cream"
```
➡ Starting text for generation.
#### Step 4: Tokenize input

Python
```
input_ids = tokenizer.encode(prompt, return_tensors="pt")
```
➡ Converts text into numerical tokens.
#### Step 5: Generate text

Python
```
output_ids = model.generate(
    input_ids,
    max_length=80,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    do_sample=True,
)
```
##### Parameters Explained:
max_length → maximum output length
temperature → randomness (lower = safer text)
top_k → limits word choices
top_p → nucleus sampling
do_sample → enables randomness
#### Step 6: Decode output

Python
```
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```
➡ Converts tokens back into readable text.
🖨️ Sample Output
```
ice cream is one of the most popular desserts enjoyed by people of all ages...
```
### Applications
Chatbots
Story generation
Creative writing
AI learning projects
NLP experimentation
### 📚 Learning Outcome
✔ Understand how pretrained language models work
✔ Learn basic text generation
✔ Get hands-on experience with Transformers
### 👩‍💻 Author
Nivetha Muralikannan
