# Requirement Classifier – ML-Based Chatbot

The **Requirement Classifier** is a machine-learning-based application designed to assist requirements engineers by automatically classifying software or system requirements.  
The system distinguishes between **functional** and **nonfunctional** requirements and further categorizes nonfunctional requirements into specific classes.

---

## 🎯 Project Goal

The goal of this project is to simplify and speed up the requirements analysis process by providing an interactive chatbot that classifies requirements using trained machine learning models.

---

## 🧱 System Architecture

The application is composed of three main components:

### 1. Next.js Web Application (chatbot-webapp)
- A lightweight chatbot-style user interface
- Allows users to input a requirement
- Sends the requirement to the backend and displays the predicted class

### 2. REST API Backend (chatbot-webapp)
- Ensures communication between the frontend and the model server

### 3. FastAPI Model Server (Fast_API_Model_Server)
- Hosts the trained machine learning models
- Performs inference and returns classification results
- Implemented in Python to support ML libraries and serialized `.pkl` models

A Python virtual environment (`venv`) is included to ensure consistent dependency management for all contributors.

---

## 🤖 Model Training & Classification Strategy

### Dataset
The strating training data (`Fast_API_Model_Server/data/new_dataset.csv`) is a CSV file containing:
- Project ID
- Requirement text
- Requirement class

### Feature Extraction
- Text data is transformed using the **TF-IDF** algorithm
- This captures the importance of words relative to the entire dataset

### Two-Stage Classification
To improve classification accuracy, a two-step approach is used:

1. **Binary Classification**
   - Functional (F)
   - Nonfunctional (NF)

2. **Multiclass Classification (Nonfunctional Only)**
   - Availability
   - Look and Feel
   - Performance
   - Operability
   - Security
   - Usability

Only nonfunctional requirements are used to train the second-stage classifier.

---

## 📈 Model Evaluation

Model performance was evaluated using a combination of **train–test splitting** and **cross-validation**:

- The dataset was split into **80% training** and **20% testing**
- **Cross-validation** was applied on the full dataset to ensure model stability and reduce overfitting

The following evaluation metrics were used:
- **Accuracy** 
- **Precision** 
- **Recall** 
- **F1-score** 
- **Support** 

Additionally, **confusion matrices** were analyzed to better understand class-level misclassifications, particularly among closely related nonfunctional requirement categories.

---

## 🧠 Models & Voting Strategy

The application supports three machine learning models:
- Support Vector Machine (SVM)
- Logistic Regression
- Stochastic Gradient Descent (SGD)

Users can:
- Select a single model for classification
- Use an ensemble voting strategy

**Voting logic:**
- If at least two models agree, the common prediction is selected
- If all three models disagree, the prediction from the model with the highest validation accuracy is used

---

## 📊 Dataset Statistics

| Class | Count |
|------|------|
| Functional (F) | 444 |
| Nonfunctional (NF) | 511 |
| Availability (A) | 82 |
| Look and Feel (LF) | 60 |
| Operability (O) | 77 |
| Performance (PE) | 82 |
| Security (SE) | 125 |
| Usability (US) | 85 |
| **Total** | **955** |

---

## ⚠️ Limitations

- The dataset size is relatively small
- Class imbalance exists among some nonfunctional categories

---

## 🚀 Future Improvements

- Expand the dataset with additional projects
- Introduce weighted or soft voting strategies
- Create a posbility for the user to send more then 1 requirement at a time

---

## 📌 Technologies Used

- Next.js
- FastAPI
- Python
- scikit-learn
- TF-IDF
- Numpy
- Pandas
