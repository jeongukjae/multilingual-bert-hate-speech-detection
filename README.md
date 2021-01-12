# multilingual-bert-hate-speech-detection

Multilingual hate speech detection using multilingual Bert (from tfhub)

## Model Used

Model used:

* Multilingual BERT from tfhub
  * <https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/2> (Preprocessing)
  * <https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3> (BertModel)

## Used datasets

* <https://github.com/Hironsan/HateSonar> - MIT License
* <https://github.com/kocohub/korean-hate-speech> - CC-BY-SA-4.0 License
* <https://github.com/HKUST-KnowComp/MLMA_hate_speech> - MIT License
* <https://github.com/xiaoleihuang/Multilingual_Fairness_LREC> - Apache-2.0 License
* <https://github.com/ZIZUN/korean-malicious-comments-dataset> - MIT License

I simplified this problem to binary classification (hate/non-hate) because many datasets (including the biggest one) are labeled binary and labeling criteria are ambiguous.

## Stats

* `1`: hate speech
* `0`: normal speech

### # of examples by languages

```text
english
   1: 62643
   0: 56511

spanish
   1: 1918
   0: 2913

france
   1: 3193
   0: 821

arabic
   1: 2438
   0: 915

itailian
   1: 1106
   0: 4565

portuguese
   1: 379
   0: 1473

korean
   1: 9945
   0: 8422

polish
   1: 977
   0: 9942
```

total example: `171901`

## Result

`val_loss: 0.3421 - val_acc: 0.8547 - val_f1: 0.8546`
