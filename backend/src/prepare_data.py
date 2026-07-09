import os
import kagglehub
import pandas as pd


def download_dataset():
    # Download latest version
    path = kagglehub.dataset_download("bintuulugbek/cleaned-blog-authorship-corpus")
    return pd.read_csv(path + "/dataset.csv")

def prepare_dataset():
    raw_data = download_dataset()
    data = raw_data[['text', 'age_group']].copy()

    mapping = {
        "Group1 (13–17)": 0,
        "Group2 (18–29)": 1,
        "Group3 (30–48)": 2
    }

    data['age_group'] = data['age_group'].map(mapping)
    data.to_csv('./data/prepared.csv', index=False)

def main():
    if not(os.path.exists("./data/prepared.csv")):
        prepare_dataset()


if __name__ == "__main__":
    main()