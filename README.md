# README.md

# Aspect-Based Sentiment Analysis (ABSA) using BERT + CRF

> A Two-Stage Deep Learning Pipeline for Fine-Grained Sentiment Analysis of Laptop Product Reviews using **BERT**, **Conditional Random Fields (CRF)**, and **Transformer-based Sentiment Classification**.

---

# Author

**Ujjwal Prakash**

---

# Table of Contents

* Introduction
* Problem Statement
* Motivation
* Objectives
* Project Overview
* System Architecture
* Dataset
* Technologies Used
* Model Pipeline

  * Stage 1 – Aspect Term Extraction
  * Stage 2 – Aspect Sentiment Classification
* Data Preprocessing
* Model Training
* Evaluation Metrics
* Experimental Results
* Output Examples
* Visualizations
* Installation
* Usage
* Project Structure
* Future Improvements
* Applications
* Conclusion

---

# Introduction

Customer reviews have become one of the most valuable sources of product feedback. Every day, thousands of reviews are posted on e-commerce websites such as Amazon, Flipkart, and Best Buy. While these reviews provide valuable insights, manually analyzing them is nearly impossible due to their massive volume.

Traditional sentiment analysis predicts only the **overall sentiment** of a review.

Example:

> "The display is amazing, but the battery drains quickly."

A traditional sentiment classifier labels the entire sentence as either **Positive** or **Negative**, ignoring the fact that the review contains different opinions about different product features.

Aspect-Based Sentiment Analysis (ABSA) solves this limitation by identifying:

* **What feature is being discussed?** (Aspect Extraction)
* **What opinion is expressed about that feature?** (Sentiment Classification)

Instead of assigning a single sentiment to an entire review, ABSA produces feature-level insights.

Example:

| Aspect  | Sentiment |
| ------- | --------- |
| Display | Positive  |
| Battery | Negative  |

This project implements a modern **two-stage Transformer-based ABSA pipeline** using BERT and Conditional Random Fields.

---

# Problem Statement

Given a laptop review,

> "The keyboard feels great but the battery life is disappointing."

the system should automatically detect:

| Aspect       | Sentiment |
| ------------ | --------- |
| Keyboard     | Positive  |
| Battery Life | Negative  |

The objective is to build an accurate and interpretable pipeline capable of identifying multiple aspects within a sentence and determining the sentiment associated with each aspect.

---

# Motivation

Most existing sentiment analysis systems only determine whether an entire review is positive or negative.

However, customers often express mixed opinions within the same sentence.

Understanding opinions at the aspect level enables businesses to:

* Improve specific product components
* Identify recurring issues
* Understand customer preferences
* Generate automatic review summaries
* Make data-driven product decisions

---

# Objectives

The primary objectives of this project are:

* Build an Aspect Term Extraction model
* Build an Aspect Sentiment Classification model
* Combine both models into an end-to-end ABSA pipeline
* Generate structured outputs
* Produce review analytics
* Visualize sentiment distribution
* Create Amazon-style summaries

---

# Project Overview

The project is divided into two independent stages.

```
                 Product Review
                        │
                        ▼
         Stage 1: Aspect Extraction
             (BERT + CRF Model)
                        │
                        ▼
              Extracted Aspect Terms
                        │
                        ▼
      Stage 2: Sentiment Classification
              (Fine-Tuned BERT)
                        │
                        ▼
       Aspect + Sentiment Predictions
                        │
                        ▼
   Reports • CSV • Analytics • Summaries
```

The modular design allows each stage to be trained independently, making the pipeline easier to optimize and maintain.

---

# System Architecture

```
                     Input Review
                           │
                           ▼
                Text Preprocessing
                           │
                           ▼
                 BERT Tokenizer
                           │
                           ▼
          BERT Contextual Embeddings
                           │
                           ▼
                     CRF Decoder
                           │
                           ▼
              Extracted Aspect Terms
                           │
                           ▼
        BERT Sentiment Classification
                           │
                           ▼
             Positive / Negative / Neutral
                           │
                           ▼
              Final ABSA Output
```

---

# Dataset

The project uses the **Laptop Reviews Dataset**, containing manually annotated laptop reviews.

Each record consists of:

* Review sentence
* Aspect term
* Sentiment polarity
* Character offsets

Example:

| Sentence                  | Aspect       | Sentiment |
| ------------------------- | ------------ | --------- |
| Battery life is excellent | Battery Life | Positive  |
| Keyboard is uncomfortable | Keyboard     | Negative  |

---

# Technologies Used

| Technology                | Purpose                 |
| ------------------------- | ----------------------- |
| Python                    | Programming Language    |
| PyTorch                   | Deep Learning Framework |
| Hugging Face Transformers | BERT Models             |
| CRF                       | Sequence Labeling       |
| Pandas                    | Data Processing         |
| NumPy                     | Numerical Operations    |
| Matplotlib                | Visualization           |
| Scikit-learn              | Evaluation Metrics      |
| Jupyter Notebook          | Development Environment |

---

# Model Pipeline

## Stage 1 – Aspect Term Extraction

Aspect extraction is formulated as a **sequence labeling** problem.

Each token is assigned one of three BIO tags:

| Tag   | Meaning             |
| ----- | ------------------- |
| B-ASP | Beginning of Aspect |
| I-ASP | Inside Aspect       |
| O     | Outside Aspect      |

Example:

Sentence:

```
Battery life is amazing
```

Tagged sequence:

```
Battery      B-ASP
life         I-ASP
is           O
amazing      O
```

### Why BERT?

BERT captures bidirectional contextual information, allowing it to understand relationships between words more effectively than traditional word embeddings.

### Why CRF?

Although BERT predicts token labels independently, adjacent token labels are often correlated.

The CRF layer models these dependencies, ensuring valid BIO tag transitions and improving aspect boundary detection.

---

## Stage 2 – Aspect Sentiment Classification

Once aspects are extracted, each aspect is paired with its original review and passed to a second BERT model.

Input:

```
Sentence:
Battery life is amazing.

Aspect:
Battery life
```

Output:

```
Positive
```

Possible sentiment classes:

* Positive
* Negative
* Neutral

---

# Data Preprocessing

The preprocessing pipeline includes:

* Text cleaning
* Whitespace normalization
* Character span validation
* Tokenization using BERT Tokenizer
* Offset mapping generation
* BIO label creation
* Dataset splitting
* Input encoding

---

# Model Training

### Aspect Extraction

* Model:

  * BERT Base Uncased
  * CRF Decoder

Training configuration:

* Optimizer: AdamW
* Learning Rate: 3e-5
* Batch Size: 16
* Epochs: 4
* Weight Decay: 0.01

---

### Sentiment Classification

Model:

* BertForSequenceClassification

Training configuration:

* Optimizer: AdamW
* Learning Rate: 2e-5
* Epochs: 6
* Batch Size: 16

---

# Evaluation Metrics

Aspect Extraction:

* Precision
* Recall
* Exact Match F1
* Overlap F1

Sentiment Classification:

* Accuracy
* Precision
* Recall
* Macro F1-score

Pipeline:

* End-to-End Precision
* End-to-End Recall
* Overall F1-score

---

# Experimental Results

The implemented pipeline achieved strong performance across both stages.

### Aspect Extraction

* Exact Match F1 ≈ **0.81**
* Overlap F1 ≈ **0.91**

### Sentiment Classification

* Validation Accuracy ≈ **77.5%**
* Macro F1 ≈ **0.74**

### Complete Pipeline

* Precision ≈ **0.92**
* Recall ≈ **0.96**
* Overall F1 ≈ **0.94**

These results demonstrate that the proposed BERT + CRF architecture effectively extracts aspect terms while maintaining high sentiment classification accuracy.

---

# Output Example

Input Review

```
The keyboard is comfortable, but the battery drains quickly.
```

Prediction

| Aspect   | Sentiment |
| -------- | --------- |
| Keyboard | Positive  |
| Battery  | Negative  |

---

# Visualizations

The notebook generates multiple analytical visualizations, including:

* Training Loss Curves
* Aspect Extraction Performance
* Sentiment Distribution
* Confusion Matrix
* CRF Transition Matrix
* Pipeline Metrics
* Most Praised Aspects
* Most Criticized Aspects

These visualizations provide deeper insights into both model performance and customer feedback trends.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/ABSA-BERT-CRF.git
```

Move into the project directory:

```bash
cd ABSA-BERT-CRF
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

# Usage

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```
ABSA.ipynb
```

Run all cells sequentially.

The notebook performs:

* Dataset loading
* Data preprocessing
* Model training
* Evaluation
* Prediction
* Visualization
* Summary generation

---


# Future Improvements

Potential enhancements include:

* Multilingual Aspect-Based Sentiment Analysis
* Cross-domain adaptation
* Implicit aspect detection
* Knowledge Graph integration
* Large Language Model (LLM) based summarization
* Real-time deployment using FastAPI or Flask
* Web application interface
* Cloud deployment with Docker and Kubernetes

---

# Applications

This project can be applied in various domains, including:

* E-commerce review analysis
* Product recommendation systems
* Customer feedback analytics
* Business intelligence dashboards
* Brand monitoring
* Market research
* Opinion mining
* Social media analytics

---

# Conclusion

This project presents a robust and interpretable **two-stage Aspect-Based Sentiment Analysis pipeline** that combines the contextual understanding of **BERT** with the structured sequence modeling capabilities of **Conditional Random Fields (CRF)**. By separating aspect extraction from sentiment classification, the architecture achieves modularity, flexibility, and strong predictive performance.

The system accurately identifies explicit product aspects, determines their associated sentiment, and produces structured outputs suitable for real-world analytics. Beyond prediction, it provides meaningful visualizations, aspect-level insights, and automated summaries that bridge technical model outputs with practical business applications.

With its scalable design and strong end-to-end performance, this implementation serves as a solid foundation for advanced opinion mining systems and can be extended to multilingual, cross-domain, and production-ready sentiment analysis solutions.

---

## ⭐ If you found this project useful, consider giving the repository a star!
