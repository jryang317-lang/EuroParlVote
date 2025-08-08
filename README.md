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

