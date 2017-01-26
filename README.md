MNISTer
====

## Description
mnist 推論、訓練用の画像データ（28 x 28）をマウスによる手書きで作成するアプリです。

## Requirement
- Python 2.7.x
- numpy
- PyQt4
- OpenCV (cv2 モジュール)

## Usage
アプリの起動  
`$ python mnister.py`

アプリの使い方
- 右の四角に数字を描く
- Judge ボタンを押す
- 左の四角にダウンサンプリングされた画像（28 x 28）が表示される
- 下の四角に推論結果が表示される
- 推論結果が合っていれば Correct ボタンを押す
- 間違っていれば正しい結果をコンボボックスから選択し、 Submit ボタンを押す
- ダウンサンプリングされた画像が保存される

## Install
任意のディレクトリに clone してください。

## ToDo
- 手書き入力と同時に訓練もできるようにする
- ネットワークと学習用パラメータファイルの選択を可能にする
- 起動時に画像保存用フォルダを自動で作成する

## References
1. [Painting on a Widget](https://www.codeproject.com/Articles/373463/Painting-on-a-Widget "Qt での手書き文字入力")
2. [『ゼロから作る Deep Learning』のリポジトリ](https://github.com/oreilly-japan/deep-learning-from-scratch)

## Licence
Copyright (c) 2017 matsu490
Released under the MIT license
https://github.com/matsu490/MNISTer/blob/master/LICENSE.txt

## Author
[matsu490](https://github.com/matsu490)
