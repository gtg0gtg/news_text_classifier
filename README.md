📰 News Text Classifier (NLP + Scikit-Learn)

A machine learning model for classifying short English news headlines into 5 categories:

disaster

economy

health

politics

sports

The project is built end-to-end using TF-IDF + Logistic Regression.

=================================================================

Project Structure : 

news_text_classifier/
│
├── data/
│   ├── disaster.txt
│   ├── economy.txt
│   ├── health.txt
│   ├── politics.txt
│   └── sports.txt
│
├── model/
│   └── news_text_classifier.joblib
│
├── src/
│   ├── dataset_text.py
│   ├── train_text.py
│   └── predict_text.py
│
└── README.md

================================================================

🧠 Model Details

Vectorizer: TfidfVectorizer (unigram + bigram)

Classifier: LogisticRegression with class_weight="balanced"

Train/Test Split: 80/20

Metrics: Accuracy + Precision + Recall + F1


===============================================================

📈 Results (Test Set):
Using ~234 manually-collected headlines:

Accuracy ~66%

              precision    recall  f1-score   support

disaster        0.64      0.70      0.67        10
economy         0.86      0.60      0.71        10
health          0.80      0.50      0.62         8
politics        0.57      0.44      0.50         9
sports          0.59      1.00      0.74        10


========================================================

🏋️ Training:
From inside src/:

python3 train_text.py

This will:

Load the text data

Preprocess and vectorize

Train Logistic Regression

Evaluate on test set

Save the model to: ../model/news_text_classifier.joblib

========================================================

🔍 Inference (Predict)

Run prediction on any headline:

python3 predict_text.py "Global markets rise as inflation slows"

Output example:

Predicted label: economy
Confidence: 0.73

=========================================================

🛠 Requirements

Create requirements.txt:
scikit-learn
pandas
joblib

Install: 
pip install -r requirements.txt

========================================================

🚀 Future Improvements

Use more advanced ML models (SVM / LinearSVC)

Collect a larger dataset

Upgrade to transformer-based models (DistilBERT / BERT)

Add Streamlit or FastAPI demo

=======================================================

👤 Author

Qusai Ayyad
AI Engineer (in progress)
GitHub: https://github.com/gtg0gtg



