# tf.keras_imdb_reviews_Sentiment_Analysis
tf.Kerasを用いてネガティブ、ポジティブを識別する感情分析モデルを作成する。
データセットはtensorflow_datasetsの[imdb_reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews)を使用する。


## Description
コードの流れ
- データセットを読み込む
- データセットを学習、テストデータセットに分割する
- 分割したデータセットに対して前処理を行う
- ベースとなるモデルを構築、学習、学習過程を可視化する
- 過学習対策を行ったモデルを構築、学習する
- ベースモデルと過学習対策を行ったモデルのaccuracy、lossを可視化する
- ベースモデルと過学習対策を行ったモデルの評価指標を求める
- モデルが誤ってネガティブと予測したテキストをサンプリングし、予測できなかったテキストの傾向を調べる
- サンプリングしたテキストをもとに頻出単語を求めて可視化する
- 感情スコアを算出する

## Requirements
- [Python](https://www.python.org/) 3.8
- [TensorFlow](https://www.tensorflow.org/api_docs/python/tf?hl=ja)2.8.0
- [keras](https://keras.io/ja/)2.8.0
- [Numpy](http://www.numpy.org/) 1.19.2
- [sklearn](https://scikit-learn.org/stable/)1.0.2
- [matplotlib]()3.2.2

## Author

[kkkyyy2000](https://github.com/kkkyyy2000)
