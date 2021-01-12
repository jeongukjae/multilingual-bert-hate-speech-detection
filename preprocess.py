import csv
import os
import re

"""
Preprocess HateSonar
"""

with open("./HateSonar/data/labeled_data.csv", encoding="utf8") as hate_sonar_data:
    next(hate_sonar_data)

    # tuple (id, count, hate_speech, offensive_language, neither, class, tweet)
    rows_in_hate_sonar = [line for line in csv.reader(hate_sonar_data)]


def preprocess_hate_sonar(text: str):
    text = re.sub("^!+ RT", "", text)  # remove retweet
    text = re.sub("@[A-Za-z0-9_]+:?", "", text)  # remove user id
    text = re.sub("&[^;]+;", "", text)  # remove emojis and special tokens
    text = re.sub("!!+", "", text)
    text = text.replace('"', "")
    text = text.strip()
    text = re.sub(" +", " ", text)
    return text


def validate_hate_sonar(text: str):
    return "http://.+" not in text


hate_sonar_data = [
    (
        preprocess_hate_sonar(line[-1]),
        1 - int(line[2] == "0" and line[3] == "0"),
        # 1 if hate speech (offensive or hateful)
    )
    for line in rows_in_hate_sonar
    if validate_hate_sonar(line)
]


with open("hate_sonar.csv", "w", encoding="utf8") as out_file:
    writer = csv.writer(out_file)
    for row in hate_sonar_data:
        writer.writerow(row)


"""
Preprocess Korean Hate Speech
"""


def read_korean_hate_speech_dataset(tag: str):
    # tag = (train, dev)

    with open("./korean-hate-speech/labeled/" + tag + ".tsv", encoding="utf8") as labeled_file, open(
        "./korean-hate-speech/news_title/" + tag + ".news_title.txt", encoding="utf8"
    ) as news_title_file:
        next(labeled_file)

        labeled = [line for line in csv.reader(labeled_file, delimiter="\t")]
        news_title = [line for line in csv.reader(news_title_file)]

        data = [
            (
                title[0],
                comment[0],
                1 - int(comment[1] == "False" and comment[2] == "none" and comment[3] == "none"),
            )
            for title, comment in zip(news_title, labeled)
        ]

    return data


korean_hate_speech_data = read_korean_hate_speech_dataset("train") + read_korean_hate_speech_dataset("dev")

with open("korean_hate_speech.csv", "w", encoding="utf8") as out_file:
    writer = csv.writer(out_file)
    for row in korean_hate_speech_data:
        writer.writerow(row)

"""
Preprocess MLMA Hate Speech
"""

all_sentiment = set()
all_directness = set()
all_annotator_sentiment = set()
for filename in os.listdir("./hate_speech_mlma"):
    filename = os.path.join("./hate_speech_mlma", filename)

    with open(filename, encoding="utf8") as f:
        next(f)
        lines = [line for line in csv.reader(f)]
        all_sentiment |= {line[2] for line in lines}
        all_directness |= {line[3] for line in lines}
        all_annotator_sentiment |= {line[4] for line in lines}

print(all_sentiment)
print(all_directness)
print(all_annotator_sentiment)

for org_filename in os.listdir("./hate_speech_mlma"):
    filename = os.path.join("./hate_speech_mlma", org_filename)

    with open(filename, encoding="utf8") as f:
        next(f)
        data = [(line[1], int(line[2] != "normal")) for line in csv.reader(f)]

    with open("hate_speech_mlma_" + org_filename, "w", encoding="utf8") as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow(line)


"""
Preprocess Multilingual_Fairness_LREC
"""

languages = os.listdir("Multilingual_Fairness_LREC/data/split")

for lang in languages:
    data = []
    for tag in ["train", "test", "valid"]:
        with open(
            "Multilingual_Fairness_LREC/data/split/" + lang + "/" + tag + ".tsv",
            encoding="utf8",
        ) as f:
            next(f)

            lines = [line for line in csv.reader(f, delimiter="\t")]
            data += [(line[2], int(line[-1])) for line in lines]

    with open("multilingual_fairness_lrec_" + lang + ".csv", "w", encoding="utf8") as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow(line)


"""### Korean Malicious Comments"""

# !head korean-malicious-comments-dataset/Dataset.csv

with open("korean-malicious-comments-dataset/Dataset.csv", encoding="utf8") as f:
    next(f)

    lines = [line for line in csv.reader(f, delimiter="\t")]
    lines = [(line[0].split("\t") if len(line) != 2 else line) for line in lines]
    data = [(line[0], 1 - int(line[1])) for line in lines]

with open("korean-malicious-comments.csv", "w", encoding="utf8") as f:
    writer = csv.writer(f)
    for line in data:
        writer.writerow(line)
