# kdv-qt
CUDAを使用してKdV方程式を解く

![kdv-qt](https://github.com/cnloni/kdv-qt/raw/master/results/fig1.png)

## 内容
このプロジェクトは、Qiitaの記事[「CUDAでKdV方程式を解いてみる」](https://qiita.com/cnloni/items/a9826dc961401dbd2e08)に提示したプログラムの完全版である。実行形式（kdv-qt）は時刻Tにおける解のスナップショットを計算する。プログラム中で使用している数値解法およびパラメータは、N.J.Zabuskyと M.D.Kruska[(1)]の論文に記載されているものを、ほぼそのまま踏襲している.  

## 必要な要件
次のソフトウェアは、次の環境で動作確認を行っている。
+ Debian 9.4
+ CUDA 9.1

冒頭の図（Qiitaの記事における図1）を作成するには、次の要件が必要である。
+ Python 3+
+ MatplotLib 2+
+ NumPy

## インストール
```bash
# コンパイルするには
$ make

# 冒頭の図を作成するには
$ make fig1
```

## 使い方
```bash
$ bin/kdv-qt -N <the value of N> -d <the value of dt> -T < the value of T>
```
ここで、N は位置方向のデータ点の総数、dt は時間方向のきざみ、T 計算終了の時刻である。

## Files
```
kdv-qt/
+-- LICENSE
+-- README.md
+-- makefile
+-- bin/
|   +-- kdv-qt   
+-- results/
|   +-- fig1.png
|   +-- fig1.py
+-- src/
    +-- CnlArray.h
    +-- KdV.cu
    +-- KdV.h
    +-- KdVDevice.cu
    +-- gb-kdv.cpp
```

### bin/
実行形式（kdv-qt）を格納するディレクトリ。自動的に作成する。

### results/
計算結果を格納するディレクトリ。

### src/
ソース（\*.cu、\*.cpp、\*.h files）を格納するディレクトリ。

## ライセンス
ApacheライセンスV2です。./LICENCEに記述します。

Copyright (c) 2018 HIGASHIMURA Takenori <oni@cynetlab.com>

[(1)]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.15.240
