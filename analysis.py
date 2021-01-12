import csv
import os
import collections

datasets = [filename for filename in os.listdir(".") if filename.endswith(".csv")]

dataset_to_lang = {
    "multilingual_fairness_lrec_English.csv": "english",
    "hate_speech_mlma_en_dataset_with_stop_words.csv": "english",
    "multilingual_fairness_lrec_Spanish.csv": "spanish",
    "hate_speech_mlma_fr_dataset.csv": "france",
    "hate_speech_mlma_ar_dataset.csv": "arabic",
    "hate_sonar.csv": "english",
    "multilingual_fairness_lrec_Italian.csv": "itailian",
    "multilingual_fairness_lrec_Portuguese.csv": "portuguese",
    "korean_hate_speech.csv": "korean",
    "korean-malicious-comments.csv": "korean",
    "hate_speech_mlma_en_dataset.csv": "english",
    "multilingual_fairness_lrec_Polish.csv": "polish",
}

lang_count = collections.defaultdict(lambda: {"1": 0, "0": 0})

for dataset, lang in dataset_to_lang.items():
    with open(dataset, encoding="utf8") as f:
        labels = [line[-1] for line in csv.reader(f)]
        lang_count[lang]["1"] += len([l for l in labels if l == "1"])
        lang_count[lang]["0"] += len([l for l in labels if l == "0"])

for key, value in lang_count.items():
    print(key)
    print("   1: " + str(value["1"]))
    print("   0: " + str(value["0"]))
    print()
