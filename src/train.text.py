import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, label_ranking_average_precision_score
from sklearn.pipeline import Pipeline
from torchmetrics.functional import confusion_matrix

from dataset_text import load_text_data

def main():

    df = load_text_data("../data")
    print("Total samples:", len(df))
    print(df.head())

    print("\nLabel distribution:")
    print(df["label"].value_counts())

    x = df["text"]
    y = df["label"]


    # train / test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)

    # pipeline: TF-IDF + Logistic Regression
    pipline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=3000,
            ngram_range=(1,2),
            stop_words="english",

        )),
        ("clf", LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            class_weight="balanced",
        ))
    ])

    # train

    print("\nTraining model...")
    pipline.fit(x_train, y_train)

    # vald
    y_pred = pipline.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    print("classification report:")
    print(classification_report(y_test, y_pred))

    import joblib

    os.makedirs("../model", exist_ok=True)
    joblib.dump(pipline, "../model/news_text_classifier.joblib")
    print("\nSaving model...")


    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred,task="multiclass")

    plt.figure(figsize=[10,5])
    sns.heatmap(cm, annot=True,fmt="d",
                xticklabels=df["label"].unique(),
                yticklabels=df["label"].unique())

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
