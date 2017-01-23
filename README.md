MNISTer
====

Overview

## Description
mnist 推論、訓練用の画像データ（28 x 28）をマウスによる手書きで作成するアプリです。

## Requirement
python 2.7.x
numpy
PyQt4
OpenCV (cv2 モジュール)

## Usage
アプリの起動
$ python mnister.py

アプリの使い方
・右の四角に数字を描く
・Judge ボタンを押す
・左の四角にダウンサンプリングされた画像（28 x 28）が表示される
・下の四角に推論結果が表示される
・推論結果が合っていれば Correct ボタンを押す
・間違っていれば正しい結果をコンボボックスから選択し、 Submit ボタンを押す
・ダウンサンプリングされた画像が保存される

## Install
任意のディレクトリに clone してください。

## Licence
未定

## Author
[matsu490](https://github.com/matsu490)
