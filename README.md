# Arabic News Fine-Tuning with Qwen LLM

## Project Overview

This project demonstrates the **fine-tuning of a pre-trained Qwen large language model (LLM)** on **Arabic news data**, leveraging modern NLP tooling and scalable inference solutions. The primary objective is to **adapt a multilingual LLM** for **Arabic-specific tasks** such as news understanding, entity extraction, and summarization.

---

## Goal

* Customize a base LLM (Qwen) for **Arabic news comprehension** using **supervised fine-tuning (SFT)**.
* Evaluate its ability to handle **downstream tasks** in Arabic (NER, translation).
* Optimize and deploy the model for **fast inference using vLLM**.
* Offer an **educational guide** and practical workflow for parameter-efficient fine-tuning (LoRA).

---

## Tools & Libraries Used

| Tool                           | Purpose                                 |
| ------------------------------ | --------------------------------------- |
| **Hugging Face Transformers**  | Model loading and training              |
| **LLaMA-Factory**              | Training orchestration and LoRA support |
| **vLLM**                       | Optimized inference backend             |
| **Weights & Biases**           | Experiment tracking                     |
| **Pydantic**                   | Schema validation for data              |
| **Google Colab**               | Training environment                    |
| **Qwen-7B**                    | Base pre-trained model                  |
| **LoRA (Low-Rank Adaptation)** | Parameter-efficient fine-tuning         |

---

## Project Structure

```
llm_finetuning/
│
├── llm_finetuning.ipynb    # Full Colab notebook
├── dataset/                 # Arabic news dataset (user-provided)
├── outputs/                 # Model checkpoints, logs, etc.
└── README.md                # Project overview
```

---

## Workflow Summary

### 1. Environment Setup

* Installation of required libraries via `pip`.
* Hugging Face, LLaMA-Factory, and vLLM configuration.

### 2. Data Preparation

* Dataset converted to structured JSON using **Pydantic** models.
* Format: Instruction-style prompts with input-output pairs.

### 3. Model Fine-Tuning (SFT)

* Based on **Qwen** LLM with LoRA adapters for efficient training.
* Configured and launched with `LLaMA-Factory`.
* Tracked on **WandB** for monitoring.

### 4. Evaluation

* Custom prompts used to validate capabilities:

  * Summarization
  * Named Entity Recognition
  * Translation

### 5. Inference with vLLM

* Model served with **vLLM** for ultra-fast inference.
* Cost-effective token usage validated.

### 6. Token Cost Estimation

* Calculated expected cost of using the fine-tuned model per 1K tokens.

---

## Results

| Task            | Result                                                   |
| --------------- | -------------------------------------------------------- |
| Summarization   | Accurately condenses long Arabic news articles           |
| NER             | Extracts named entities with high precision              |
| Translation     | Maintains fluency and meaning between Arabic and English |
| Inference Speed | Significantly faster via vLLM vs traditional pipelines   |

The fine-tuned model demonstrates **enhanced understanding** of Arabic language nuances in real-world news.

---

## Author

Made with ❤️ by **Charif El Belghiti**

---

## 📚 Useful Resources

* [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)  
* [Qwen Model on Hugging Face](https://huggingface.co/Qwen)  
* [vLLM: Fast Inference Engine](https://github.com/vllm-project/vllm)  
* [LoRA Paper](https://arxiv.org/abs/2106.09685)

---

