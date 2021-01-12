import csv
import os
import copy
import random

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
import tensorflow_text as text

from tqdm import tqdm

datasets = [filename for filename in os.listdir(".") if filename.endswith(".csv")]

dataset_file_to_lang = {
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

print(set(dataset_file_to_lang.values()))

dataset_by_lang = {
    "polish": [],
    "korean": [],
    "portuguese": [],
    "arabic": [],
    "france": [],
    "spanish": [],
    "itailian": [],
    "english": [],
}

for filename, lang in dataset_file_to_lang.items():
    with open(filename, encoding="utf8") as f:
        dataset_by_lang[lang].extend(
            [
                (line[0], int(line[1])) if len(line) == 2 else ((line[0], line[1]), int(line[2]))
                for line in csv.reader(f)
            ]
        )

train_dataset_by_lang = {
    "polish": [],
    "korean": [],
    "portuguese": [],
    "arabic": [],
    "france": [],
    "spanish": [],
    "itailian": [],
    "english": [],
}
dev_dataset_by_lang = copy.deepcopy(train_dataset_by_lang)

for lang in dataset_by_lang.keys():
    random.shuffle(dataset_by_lang[lang])
    length = len(dataset_by_lang[lang])
    dev_length = int(length * 0.1)
    dev_dataset_by_lang[lang] = dataset_by_lang[lang][:dev_length]
    train_dataset_by_lang[lang] = dataset_by_lang[lang][dev_length:]

for lang in dataset_by_lang.keys():
    print(lang)
    print("  train: " + str(len(train_dataset_by_lang[lang])))
    print("  dev: " + str(len(dev_dataset_by_lang[lang])))

train_set = [data for data_by_lang in train_dataset_by_lang.values() for data in data_by_lang]
dev_set = [data for data_by_lang in dev_dataset_by_lang.values() for data in data_by_lang]

print(len(train_set), len(dev_set))

preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2")
tokenize = hub.KerasLayer(preprocessor.tokenize)
bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=128))
single_bert_input = hub.KerasLayer(preprocessor)


def take_first(item):
    return {
        "input_mask": item["input_mask"][0],
        "input_type_ids": item["input_type_ids"][0],
        "input_word_ids": item["input_word_ids"][0],
    }


@tf.function(input_signature=[tf.TensorSpec([], dtype=tf.string), tf.TensorSpec([], dtype=tf.int32)])
def parse_single(item, label):
    return [take_first(single_bert_input([item])), tf.one_hot(label, 2)]


@tf.function(
    input_signature=[
        tf.TensorSpec([], dtype=tf.string),
        tf.TensorSpec([], dtype=tf.string),
        tf.TensorSpec([], dtype=tf.int32),
    ]
)
def parse_multi(item1, item2, label):
    return [take_first(bert_pack_inputs([tokenize([item1]), tokenize([item2])])), tf.one_hot(label, 2)]


train_tensor_set = [
    (parse_single(item[0], item[1]) if not isinstance(item[0], tuple) else parse_multi(item[0][0], item[0][1], item[1]))
    for item in tqdm(train_set)
]

dev_tensor_set = [
    (parse_single(item[0], item[1]) if not isinstance(item[0], tuple) else parse_multi(item[0][0], item[0][1], item[1]))
    for item in tqdm(dev_set)
]

train_set = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(
            {
                "input_mask": [i[0]["input_mask"] for i in train_tensor_set],
                "input_type_ids": [i[0]["input_type_ids"] for i in train_tensor_set],
                "input_word_ids": [i[0]["input_word_ids"] for i in train_tensor_set],
            }
        ),
        tf.data.Dataset.from_tensor_slices([i[1] for i in train_tensor_set]),
    )
)
print(train_set.element_spec)

dev_set = tf.data.Dataset.zip(
    (
        tf.data.Dataset.from_tensor_slices(
            {
                "input_mask": [i[0]["input_mask"] for i in dev_tensor_set],
                "input_type_ids": [i[0]["input_type_ids"] for i in dev_tensor_set],
                "input_word_ids": [i[0]["input_word_ids"] for i in dev_tensor_set],
            }
        ),
        tf.data.Dataset.from_tensor_slices([i[1] for i in dev_tensor_set]),
    )
)
print(dev_set.element_spec)

encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3", trainable=True)


def create_model():
    input_node = {
        "input_mask": tf.keras.Input([None], dtype=tf.int32),
        "input_type_ids": tf.keras.Input([None], dtype=tf.int32),
        "input_word_ids": tf.keras.Input([None], dtype=tf.int32),
    }
    encoder_outputs = encoder(input_node)["pooled_output"]
    output_prob = tf.keras.layers.Dense(2, activation="softmax", name="output")(encoder_outputs)
    model = tf.keras.Model(input_node, output_prob)
    return model


train_set = train_set.shuffle(len(train_tensor_set), reshuffle_each_iteration=True).batch(32, drop_remainder=True)
dev_set = dev_set.batch(64)

print("total example: " + str(len(train_tensor_set)))
print("step per epoch: " + str(len(train_tensor_set) // 32))

model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adamax(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5e-5,
            decay_steps=100,
            decay_rate=0.97,
        )
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=["acc", tfa.metrics.F1Score(num_classes=2, name="f1")],
)
model.fit(
    train_set,
    epochs=3,
    validation_data=dev_set,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint("./models/weights.epoch-{epoch:02d}", verbose=1),
        tf.keras.callbacks.TensorBoard("./logs", update_freq="batch"),
    ],
)
