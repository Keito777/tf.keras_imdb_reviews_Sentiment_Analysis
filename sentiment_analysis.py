import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pprint
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import collections
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

print(tf.__version__)

#Dataset　Download（infoはメタデータ）
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

#Datasetのの概要を確認する
print(imdb)
print(len(imdb))
df = pd.DataFrame(imdb["train"])
df.head()
df.info()

#学習、テストデータに分割する
train, test = imdb["train"], imdb["test"]

train_sentences = []
train_labels = []

test_sentences = []
test_labels = []

#TensorをNumpyオブジェクトに変更して値を取得する
for s, l in train:
  train_sentences.append(str(s.numpy()))
  train_labels.append(l.numpy())
  
for s, l in test:
  test_sentences.append(str(s.numpy()))
  test_labels.append(l.numpy())

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

#padding
vocab_size = 10000
embedding_dim = 16
max_length = 120
padding_type="post"
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences,maxlen=max_length, padding=padding_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences,maxlen=max_length, padding=padding_type)

#padding前後のテキストを確認する
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])
print(f"元のテキスト：{train_sentences[1]}\n")
print(f"padding後のテキスト：{train_padded[1]}\n")
print(f"復元後のテキスト：{decode_review(train_padded[1])}")

#学習、テストデータの型とラベルを確認する
print(f"学習データSize：{train_padded.shape}")
print(f"学習データlabel_Size:{train_labels.shape}")
print(f"テストデータSize：{test_padded.shape}")
print(f"テストデータlabel_Size:{test_labels.shape}")

print(type(train_padded))
print(type(train_labels), "\n")
print(type(test_padded))
print(type(test_labels))

#Conv1D_GlobalAveragePooling1D_Model（ベースモデル）
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation="relu"),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#compile
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()

#fit
num_epochs = 5
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

#evaluate
model.evaluate(test_padded, test_labels)

#plot
acc=history.history["accuracy"]
val_acc=history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(len(acc))

plt.plot(epochs, acc, "r")
plt.plot(epochs, val_acc, "b")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()

plt.plot(epochs, loss, "r")
plt.plot(epochs, val_loss, "b")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.show()

#Bidirectional_GRU_Dropout_Model（過学習対策モデル）
model2 = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#compile
model2.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
model2.summary()

#fit
num_epochs = 5
history2 = model2.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels))

#evaluate
model2.evaluate(test_padded, test_labels)

#plotacc=history2.history["accuracy"]
val_acc=history2.history["val_accuracy"]
loss=history2.history["loss"]
val_loss=history2.history["val_loss"]
epochs=range(len(acc))

plt.plot(epochs, acc, "r")
plt.plot(epochs, val_acc, "b")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.show()

plt.plot(epochs, loss, "r")
plt.plot(epochs, val_loss, "b")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.show()

#comparison
acc=history.history["accuracy"]
val_acc=history.history["val_accuracy"]
loss=history.history["loss"]
val_loss=history.history["val_loss"]

d_acc=history2.history["accuracy"]
d_val_acc=history2.history["val_accuracy"]
d_loss=history2.history["loss"]
d_val_loss=history2.history["val_loss"]

epochs=range(len(acc))
epochs2=range(len(d_acc))

def acc_plot(epochs, epochs2, acc, val_acc, d_acc, d_val_acc):
  plt.plot(epochs, acc, "r")
  plt.plot(epochs, val_acc, "b")
  plt.plot(epochs2, d_acc, "m")
  plt.plot(epochs2, d_val_acc, "c")
  
  plt.title("Training and validation accuracy")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend(["Acc", "Valid_Acc", "Dropout_Acc", "Dropout_Val_Acc"])
  plt.show()

def loss_plot(epochs, epochs2, loss, val_loss, d_loss, d_val_loss):
  plt.plot(epochs, loss, "r")
  plt.plot(epochs, val_loss, "b")
  plt.plot(epochs2, d_loss, "m")
  plt.plot(epochs2, d_val_loss, "c")

  plt.title("Training and validation loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend(["Loss", "Validation Loss", "Dropout_Loss", "Dropout_Val_Loss"])
  plt.show()

acc_plot(epochs, epochs2, acc, val_acc, d_acc, d_val_acc)
loss_plot(epochs, epochs2, loss, val_loss, d_loss, d_val_loss)

#テストラベルの分布を確認する
test_label_num = pd.DataFrame(test_labels)
test_label_num.value_counts()

#混同行列を表示する
def make_cm(matrix, columns):
    n = len(columns)
    act = ["正解データ"] * n
    pred = ["予測結果"] * n
    cm = pd.DataFrame(matrix, 
        columns=[pred, columns], index=[act, columns])
    return cm

cm = make_cm(c_matrix2, ["NEGATIVE", "POSITIVE"])
cm

#評価指標を確認する（ベースモデル、過学習対策モデルの順に表示する）
print(classification_report(test_labels, y_pred, target_names=["0_pos", "1_pos"]), "\n")
print(classification_report(test_labels, y_pred2, target_names=["0_pos", "1_pos"]))

#Sampling
#ポジティブなテキストをモデルがネガティブと予測したテキストを取得する
samp_index = []
index = 0

for label, pred in zip(test_labels, y_pred2):
  if label == 1 and pred ==0:
    samp_index.append(index)
    index += 1
  else:
    index +=1

#複数のインデックスを指定し、要素をまとめて取得する
x_test_numpy = np.array(test_sentences)
x_test_samples = x_test_numpy[samp_index]

#ポジティブなのにネガティブと予測したテキストをDataFrameで表示する
sample_df = pd.DataFrame(x_test_samples, columns=["sample_text"])
sample_df.head(100)

#テストデータに対してstop_wordの削除とstemming、トークン化を行いデータの傾向を調査する
def filter_stop_words(sentences, stop_words):
    for i, sentence in enumerate(sentences):
        new_sent = [word for word in sentence.split() if word not in stop_words]
        sentences[i] = " ".join(new_sent)
    return sentences

#stop_wordの削除
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
sw_x_test = filter_stop_words(x_test_samples, stop_words)

#stemming
snowball = SnowballStemmer(language="english")
clean_test_words = [snowball.stem(t) for t in sw_x_test]

#トークン化
nltk.download("punkt")
w_list = []
for t in clean_test_words:
  t = nltk.word_tokenize(t)
  for w in t:
    w_list.append(w)
pd.Series(w_list).value_counts().head(30)

#plot
c = collections.Counter(w_list)
fig = plt.subplots(figsize=(8, 8))
sns.countplot(y=w_list,order=[i[0] for i in c.most_common(20)])

#感情スコアを算出する
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

#感情スコアを算出する関数を実装する
def decode_sentiment(score, include_neutral=True):
    if include_neutral:        
        label = NEUTRAL
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = NEGATIVE
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = POSITIVE
        return label
    else:
        return NEGATIVE if score < 0.5 else POSITIVE

def predict(text, model, include_neutral=True):
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_length, padding=padding_type)
    score = model.predict([x_test])
    label = decode_sentiment(score, include_neutral=include_neutral)
    results = f"label : {label}, score : {float(score):.2f}, text : {text}"
    return results
  
#感情スコア算出ようのテキストを抽出する
def text_label(text_index, label_index):
  text = test_sentences[text_index]
  label = test_labels[label_index]
  result = f"Text : {text}\nLabel : {label}"
  return result

print(text_label(1, 1))
print(text_label(2, 2))

#感情スコアを確認する
text1 = test_sentences[1] #label:1
text2 = test_sentences[2] #label:0
print(predict(text1, model2))
print(predict(text2, model2))
