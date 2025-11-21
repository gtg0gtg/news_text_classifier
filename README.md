# News Text Classifier (NLP + Scikit-Learn)

A machine learning model for classifying short English news headlines into five categories: disaster, economy, health, politics, sports. The project is built end-to-end using TF-IDF + Logistic Regression.

## Project Structure
news_text_classifier/
- data/ (each .txt file contains headlines: disaster.txt, economy.txt, health.txt, politics.txt, sports.txt)
- model/ (saved model: news_text_classifier.joblib)
- src/ (dataset_text.py, train_text.py, predict_text.py)
- README.md




## Model Details
Vectorizer: TfidfVectorizer (unigram + bigram)  
Classifier: LogisticRegression with class_weight="balanced"  
Train/Test Split: 80/20  
Metrics: Accuracy, Precision, Recall, F1-score  




## Results (Test Set)
Accuracy ≈ 66%
Disaster: f1 = 0.67  
Economy: f1 = 0.71  
Health: f1 = 0.62  
Politics: f1 = 0.50  
Sports: f1 = 0.74  

This is a strong baseline considering the dataset size (~234 headlines).




## Training
From inside src/, run:
python3 train_text.py  
This loads text data, vectorizes it, trains Logistic Regression, evaluates it, and saves the model to ../model/news_text_classifier.joblib




## Prediction
Run:
python3 predict_text.py "Global markets rise as inflation slows"  
Output example: Predicted label: economy — Confidence: 0.73  
Or run without arguments to enter text manually.




## Requirements
scikit-learn  
pandas  
joblib  
Install with: pip install -r requirements.txt




## Future Improvements
- Use SVM / LinearSVC  
- Collect larger dataset  
- Test transformer models (DistilBERT / BERT)  
- Add a Streamlit or FastAPI web demo




## Author
Qusai Ayyad  
GitHub: https://github.com/gtg0gtg
