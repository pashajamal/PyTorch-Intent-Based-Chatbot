# Intent-Based PyTorch Chatbot Assistant

An adaptable, intent-based contextual chatbot assistant built from scratch using C++ styled neural networks implemented via **PyTorch** and **NLTK**. The chatbot processes natural language inputs using tokenization, lemmatization, and a Bag-of-Words pipeline to predict intent classes, fetch corresponding responses, and dynamically trigger custom backend functions.

## 🚀 Features
* **Custom NLP Pipeline:** Text preprocessing utilizing NLTK's word tokenization and WordNet lemmatization.
* **Deep Learning Engine:** 3-layer fully connected sequential Neural Network built using PyTorch with ReLU activations and Dropout regularization layers.
* **Dynamic Function Mapping:** Automatically execute specific Python functions (e.g., fetching portfolios) tied directly to a predicted intent tag.
* **Flexible Intents Configuration:** Easily add new capabilities by adjusting a simple JSON data profile (`intents.json`).

---

## 🏗️ Architecture Summary
The deep learning module uses a standard classification model (`ChatbotModel`) structured as follows:
* **Layer 1:** Fully connected input layer -> 128 Hidden units -> ReLU -> Dropout (50%)
* **Layer 2:** 128 Hidden units -> 64 Hidden units -> ReLU -> Dropout (50%)
* **Layer 3:** 64 Hidden units -> Output classification logits matching total unique intent categories.

---

## 🛠️ Requirements & Installation

Make sure Python is installed on your local machine, then install the necessary dependencies:

```bash
pip install torch numpy nltk