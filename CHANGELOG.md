# Change Log

## [v0.3.4](https://github.com/matsu490/MNISTer/tree/v0.3.4) (2017-2-8)
**Fixed bugs:**
- 追記漏れがあったので全ファイルに coding: utf-8 を追記した

**Closed issues:**
- ValueError: insecure string pickle [\#1](https://github.com/matsu490/MNISTer/issues/1)

## [v0.3.3](https://github.com/matsu490/MNISTer/tree/v0.3.3) (2017-2-8)
**Fixed bugs:**
- ソースファイル中に日本語のコメントがあると non-ASCII エラーが出るので、ファイルに coding: utf-8 を追記した

## [v0.3.2](https://github.com/matsu490/MNISTer/tree/v0.3.3) (2017-2-8)
**Fixed bugs:**
- README を修正した

## [v0.3.1](https://github.com/matsu490/MNISTer/tree/v0.3.3) (2017-2-8)
**Implemented enhancements:**
- DeepConvNet を追加した

## [v0.3.0](https://github.com/matsu490/MNISTer/tree/v0.3.0) (2017-2-8)
**Implemented enhancements:**
- ニューラルネットワークを使って入力画像を推論できるようにした
- モデル・パラメータ選択用のダイアログを追加した
- 画像保存時に各クラス用のディレクトリを自動で作成するようにした

## [v0.2.0](https://github.com/matsu490/MNISTer/tree/v0.2.0) (2017-2-8)
**Implemented enhancements:**
- なんか色々修正した

## [v0.1.0](https://github.com/matsu490/MNISTer/tree/v0.1.0) (2017-2-8)
**Implemented enhancements:**
- 初めてのリリース
- とりあえず入力画像をダウンサンプリングして保存する機能だけ付けた

## ToDo
- [ ] 手書き入力と同時に訓練もできるようにする
- [x] ネットワークと学習用パラメータファイルの選択を可能にする
- [ ] 現在の正答率を表示する
- [x] 起動時に画像保存用フォルダを自動で作成する
- [ ] 推論結果上位3つを確率とともに表示する
