import os
import pandas as pd

def load_text_data(data_dir):

    texts = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            label = filename.replace(".txt", "")
            filepath = os.path.join(data_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            lines = [line.strip() for line in lines if line.strip()]

            for line in lines:
                texts.append(line)
                labels.append(label)

    df = pd.DataFrame({
        "text": texts,
        "label": labels

    })

    return df

if __name__ == "__main__":
    df = load_text_data("../data")
    print(df.head())
    print("total samples:", len(df))

