# 📌 Comment Toxicity Detection (Deep Learning + Streamlit)

## 📖 Project Overview  
This project builds an AI-powered system to detect toxic comments using Deep Learning (LSTM) and NLP.

It classifies comments into:
- Toxic  
- Severe Toxic  
- Obscene  
- Threat  
- Insult  
- Identity Hate  

The model is deployed using a Streamlit web application.

---

## 🎯 Problem Statement  
Online platforms face hate speech and offensive language.  
Manual moderation is slow.  
This project automates toxicity detection.

---

## 🚀 Features  
- Real-time prediction  
- Multi-label classification  
- Bulk CSV prediction  
- Streamlit UI  
- Model visualization  

---

## 🧠 Tech Stack  
Python, TensorFlow, Keras, NLTK, Pandas, NumPy, Scikit-learn, Streamlit  

---

## 📂 Project Structure  
CommentToxicity/
├── artifacts/
├── train.csv
├── test.csv
├── data.py
├── app.py
└── README.md

---

## 🔄 Workflow  
Data → Preprocessing → Tokenization → LSTM → Prediction → Streamlit  

---

## ▶️ How to Run  

### Install Dependencies  
pip install -r requirements.txt  

### Train Model  
python data.py  

### Run App  
streamlit run app.py  

---

## 📥 Input Format  
CSV must contain column: comment_text  

---

## 📊 Output  
Shows probability for each toxicity label.

---

## 🌍 Use Cases  
Social media moderation, forums, brand safety, education platforms  

---

## 📈 Future Work  
- Use BERT  
- Deploy API  
- Cloud deployment  

---

## 👨‍💻 Author  
Sai Haranadh
