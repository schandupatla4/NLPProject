# NLPProject
Project Title: Supervised In-Context Learning for Robust NLP Tasks

This project explores the application of Supervised In-Context Learning (SuperICL) for various NLP tasks using datasets from the GLUE benchmark (e.g., MNLI, SST-2, MRPC). The approach integrates pre-trained models like RoBERTa with large language models (LLMs) to enhance prediction robustness.
Features

    Fine-tuning pre-trained models on specific tasks.
    Dynamic selection of in-context examples based on cosine similarity.
    Integration of plug-in predictions with LLMs for improved performance.
    Evaluation across tasks to measure accuracy and robustness.

Notebooks
1. ORGINAL.ipynb

    Tasks Addressed: Fine-tuning and evaluation on GLUE tasks (e.g., MRPC, SST-2).
    Key Steps:
        Preprocess datasets (e.g., tokenize and structure data).
        Train and fine-tune models using Hugging Face Transformers.
        Evaluate performance using metrics like accuracy and F1-score.
    Highlights:
        Focuses on MRPC and SST-2 datasets.
        Implements a reduced context version of SuperICL for efficient evaluation.

2. Robustness.ipynb

    Tasks Addressed: Robustness analysis using MNLI and SST-2.
    Key Steps:
        Evaluate plug-in predictions with confidence thresholds.
        Integrate LLM predictions to override plug-in results when uncertainty arises.
        Calculate accuracy and explain model decisions with detailed reasoning.
    Highlights:
        Dynamic in-context example selection based on cosine similarity.
        Uses RoBERTa as a plug-in model and GPT-4 as an LLM.

Installation
Prerequisites

    Python 3.7 or higher
    Libraries:
        Transformers (Hugging Face)
        PyTorch
        Scikit-learn
        OpenAI API
        Datasets (Hugging Face)

Setup

    Clone the repository:

git clone https://github.com/schandupatla4/NLPProject.git    
cd NLPProject

Install dependencies:

pip install -r requirements.txt

Set up OpenAI API:

    Add your API key in an environment variable:

        export OPENAI_API_KEY=your_api_key

Usage
Fine-tuning

Run ORGINAL.ipynb to fine-tune pre-trained models on MRPC or SST-2 tasks.
Robustness Evaluation

Use Robustness.ipynb to analyze performance on dynamic in-context learning tasks.
Results

    SST-2 Accuracy: Achieved X% using SuperICL with RoBERTa and GPT-4.
    MRPC Accuracy: Achieved Y% with enhanced fine-tuning.

Future Work

    Expand experiments to other GLUE tasks (e.g., QNLI, CoLA).
    Explore alternative LLMs for robustness enhancement.
    Optimize training for large datasets.
