# 🧮 Mathematical AI Routing System

## 📌 Overview
This project implements an **AI-powered math assistant** that intelligently processes mathematical queries, retrieves relevant information from a knowledge base or the web, and generates **step-by-step solutions** using AI models. Designed for **students, educators, and automated tutoring platforms**, it provides reliable and structured responses.

## 🚀 Features
- **Intelligent Query Routing**: Routes math problems through a **knowledge base**, **web search**, and **AI generation**.
- **Vector Search (Qdrant)**: Stores and retrieves math problems using **semantic similarity**.
- **Web Search Integration**: Uses **Perplexity AI, Tavily, and DuckDuckGo** to find math-related answers.
- **Step-by-Step AI Solutions**: Generates structured responses using **Hugging Face AI models**.
- **Ethical Guardrails**: Prevents inappropriate or non-math-related queries.
- **User Feedback System**: Allows improvement of generated solutions.

## 🏗️ Architecture
1. **User submits a math query** via the interface.
2. **GuardrailsValidator** checks for ethical compliance.
3. **KnowledgeBase** searches for previously stored solutions.
4. If no answer exists, **WebSearchAgent** finds relevant data online.
5. If needed, **SolutionGenerator** constructs a step-by-step response.
6. **Confidence scores & source attribution** are added for transparency.

## 🔧 Setup & Installation
### **Prerequisites**
- Python 3.8+
- Dependencies in `requirements.txt`

### **Installation**
```bash
git clone https://github.com/Loki-318/Math-tutor.git 
cd math-routing-agent
pip install -r requirements.txt
