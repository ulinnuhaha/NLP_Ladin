# The First NLP Benchmark for Ladin - an Extremely Low-Resource
<div align="center">
  <a href="https://huggingface.co/datasets/ulinnuha/mcqa_ladin_italian">
    <img src="https://img.shields.io/badge/HuggingFace-MCQA Dataset-yellow" alt="Hugging Face MCQA Dataset">
  </a>
  <a href="https://huggingface.co/datasets/ulinnuha/sentiment_analysis_ladin_italian">
    <img src="https://img.shields.io/badge/HuggingFace-SA Dataset-green" alt="Hugging Face SA Dataset">
  </a>
  <a href="https://arxiv.org/abs/2509.03962">
    <img src="https://img.shields.io/badge/Paper-arXiv-red" alt="Paper Link">
  </a>
</div>
This is a repository for NLP tasks in Ladin language (Val Badia Variant), including text classification, question answering, and machine translation, using LLMs and mBERT


🔥 **NLP Benchmark dataset for Ladin language**  
🔗 **Try it on [Hugging Face](https://huggingface.co/datasets/ulinnuha)**  
📄 **Read our [paper](https://arxiv.org/abs/2509.03962)**  

## 📌 Features
✅ Supports parallel sentences between Italian and Ladin (Val Badia Varaint)  
✅ High-filtered synthetic dataset  
✅ Ready for fine-tuning in machine translation, text classification, and question-answering

# Sentiment Analysis
The sentiment analysis dataset consists of 12,511 entries (Italian and Ladin) with two labels: positive and negative. To perform this task, we utilize few-shot learning using Llama and m-BERT.

To perform the classification task using LLM, please go to:
```
FSL_llm.ipynb
```

To perform the classification task using mBERT, please go to:
```
SA_mBERT.ipynb
```
# Multiple-choice question answering (MCQA)
The MCQA dataset consists of 764 entries (Italian and Ladin) with 3-5 choices. To perform this task, we utilize few-shot learning using Llama and m-BERT.

To perform the classification task using LLM, please go to:
```
FSL_llm.ipynb
```

To perform the classification task using mBERT, please go to:
```
MCQA_mBERT.ipynb
```

# Citation
If you use our resources, please cite:
```bash
@article{my_paper,
  title={Exploring NLP Benchmarks in an Extremely Low-Resource Setting},
  author={Ulin Nuha, Adam Jatowt},
  journal={ArXiv},
  year={2025}
}


