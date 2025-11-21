import sys
import joblib

model_path = "../model/news_text_classifier.joblib"

def load_model(model_path:str = model_path):
    model = joblib.load(model_path)
    return model

def predict_text(model, text:str):

    proba = model.predict_proba([text])[0]
    classes = model.classes_

    import numpy as np

    top_idx = np.argmax(proba)
    top_class = classes[top_idx]
    top_conf = proba[top_idx]

    return top_class, top_conf


def main():

    if len(sys.argv) > 1:
        text = "".join(sys.argv[1:])
    else:
        text = input("enter a news headline:").strip()

    if not text:
        print("no text provided")
        return

    model= load_model()
    top_class, top_conf = predict_text(model, text)

    print(f"predicted label: {top_class}")
    print(f"confidence: {top_conf:.4f}")

if __name__ == "__main__":
    main()
