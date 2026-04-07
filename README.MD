# 📧 AI-Powered Email Phishing Detection System

## 🚀 Description

This project is an **AI-powered email phishing detection system** designed to intelligently identify malicious and suspicious emails using advanced **machine learning and natural language processing (NLP)** techniques.

At its core, the system leverages a **CatBoost-based classification model**, trained on engineered textual, semantic, and URL-based features, to accurately distinguish between legitimate and phishing emails. The model captures hidden patterns in email content, going beyond simple keyword matching by learning complex relationships within the data.

The application implements a complete **end-to-end AI pipeline**, including:

* Data preprocessing and cleaning
* Custom NLP-based feature extraction
* Model training and evaluation
* Real-time inference through a REST API

Unlike traditional rule-based systems, this solution relies on **data-driven intelligence**, enabling it to adapt to evolving phishing strategies and improve detection performance.

The AI model is deployed within a **Flask-based backend**, allowing real-time predictions, while a responsive frontend interface provides instant feedback with probability scores and classification results.

This project demonstrates the practical application of **artificial intelligence in cybersecurity**, combining machine learning, NLP, and web technologies into a production-ready system.

---

## 🧠 Key Features

* 🔍 Automated email analysis
* 🤖 Machine learning classification using CatBoost
* 🧾 Custom NLP feature engineering
* 🌐 REST API built with Flask
* 💻 Interactive web interface (HTML/CSS/JS)
* ⚡ Real-time feedback (legitimate vs phishing)
* 📊 Configurable probability threshold (default = 0.60)

---

## 🏗️ System Architecture

```text
Frontend (HTML/CSS/JS)
        │
        │ HTTP POST /analyze
        ▼
Backend (Flask API)
        │
        ├── Feature Extraction (NLP)
        ├── Model Loading
        ├── Prediction (CatBoost)
        └── JSON Response
        ▼
AI Model (CatBoost .pkl)
```

---

## ⚙️ Technologies Used

### 🧠 Backend & AI

* Python (3.8 – 3.11)
* CatBoost (classification algorithm)
* Scikit-learn (data splitting, evaluation metrics)
* Pandas / NumPy (data processing)

### 🧾 NLP & Text Processing

* NLTK (tokenization, stopwords)
* Regex (pattern extraction, cleaning)
* BeautifulSoup (HTML parsing in emails)
* TLDExtract (URL/domain analysis)

### 🌐 API & Server

* Flask
* Flask-CORS

### 💻 Frontend

* HTML5
* CSS3 (animations, responsive design)
* JavaScript (Fetch API, DOM manipulation)

### 🧰 Utilities

* Joblib (model persistence)
* Requests (HTTP requests, optional)

---

## 📊 Feature Engineering

### ✉️ Text-Based Features

* Text length
* Word count
* Exclamation mark count
* Question mark count
* Uppercase ratio

### 🔗 URL-Based Features

* Number of URLs
* Presence of URLs
* Shortened URLs (bit.ly, tinyurl, etc.)
* URLs containing IP addresses

### ⚠️ Semantic Features

* Suspicious keyword count (e.g., "urgent", "verify")
* Urgency score
* Threat score

---

## 🧪 AI Model

* Algorithm: **CatBoostClassifier**
* 500 decision trees
* Automatic class imbalance handling
* Feature importance analysis
* Classification threshold: **0.60**

---

## 📂 Project Structure

```text
project/
│
├── backend/
│   ├── app.py
│   ├── preprocessing.py
│   ├── train_models.py
├── models/
│   └── catboost_model.pkl
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── train_final_corrected.py
├── requirements.txt
└── README.md
```

---

## ▶️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nour-Boughalmi/email-phishing-detector.git
cd email-phishing-detector
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python backend/app.py
```

### 5. Access the application

```
http://localhost:5000
```

---

## 📈 API Example

### Request

```http
POST /analyze
Content-Type: application/json

{
  "email": "Your account has been suspended. Click here to verify."
}
```

### Response

```json
{
  "prediction": "phishing",
  "probability": 0.87
}
```

---

## 🎯 Project Objectives

* Accurately detect phishing emails using AI
* Combine NLP and machine learning techniques
* Build a full-stack AI application (Frontend + Backend + Model)
* Deliver a system usable in real-world cybersecurity scenarios

---

## 👩‍💻 Author

**Nour Boughalmi**
ICT Engineering Student | AI  Enthusiast

---

## ⭐ Contributions

Contributions are welcome. Feel free to open an issue or submit a pull request.
