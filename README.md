# EuroParlVote

Codebase for reproducing **Gender Classification** and **Vote Prediction** experiments from the **EuroParlVote** dataset — linking European Parliament debate speeches with roll-call votes and Member of European Parliament (MEP) demographics across 24 official EU languages.

📄 Dataset: [https://huggingface.co/datasets/unimelb-nlp/EuroParlVote](https://huggingface.co/datasets/unimelb-nlp/EuroParlVote)  
📜 Paper: *Demographics and Democracy: Benchmarking LLMs’ Gender Bias and Political Leaning in European Parliament* (ICNLSP 2025)

---

## 📦 Installation

We recommend using **conda** for environment setup:

```bash
# Clone this repository
git clone https://github.com/jryang317-lang/EuroParlVote.git
cd EuroParlVote

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate europarlvote
```

## 🚀 Run Guide

### 1. LLaMA Models
To run experiments with **LLaMA** models, you need to request and download weights from the official Meta repository:  
🔗 [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)  

Once approved:
1. Download the weights via Hugging Face.
2. Update the model path in scripts:

```bash
python llama3.2_vote_prediction_htv.py \
  --model_path /path/to/LLaMA/model \
  --input_file data/dev_set.csv \
  --output_file preds/vote_dev_llama3.2.json
```

### 2. GPT Models
For GPT models (e.g., gpt-3.5-turbo, gpt-4o), you must provide your OpenAI API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
python gpt_predict_vote_htv.py \
  --model gpt-4o \
  --input_file data/dev_set.csv \
  --output_file preds/vote_dev_gpt4o.json
```
Get an API key here: https://platform.openai.com/account/api-keys

### 3. Gemini Models
For Google Gemini models, set your Gemini API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"
python gemini_vote_prediction_htv.py \
  --model gemini-pro \
  --input_file data/dev_set.csv \
  --output_file preds/vote_dev_gemini.json
```
Get an API key here: https://aistudio.google.com/app/apikey
