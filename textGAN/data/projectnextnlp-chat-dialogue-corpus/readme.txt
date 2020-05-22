■　ファイル
○　readme.txt
このファイル

○　WS_PNN23_dial.pdf
2015/3に開催された言語処理学会のProject Next NLPワークショップでの対話タスクの報告資料。
データの収集方法が記載されている。

○　annotator-attribute.tsv
アノテータの属性情報

○　speaker-attribute.tsv
話者の属性情報

○　json/
学習用対話データ
init100:初期アノテーション　24人で全員が全体話を実施
rest1046:2回目のアノテーション　22人で各対話2,3人ずつ実施

○　data-readme.txt
対話データの書式について

○　annotation-manual.txt
init100,rest1046のアノテーション時にアノテータに配布したマニュアル

○　LICENSE.txt
このデータのライセンスについて

○　show_dial.py
対話データを表示するpythonスクリプト
Python 2.7.9で動作確認済み
usage :
python show_dial.py JSONファイル

example : 
python show_dial.py 1408001670log.json
dialogue-id : 1408001670
speaker-id : 04_05
group-id : subset_f
S:こんにちは！気軽に声かけてね。 O O
U:はろはろ
S:はろー O O
U:うん
S:待ってる T T
U:え
S:うぇ X T
U:うぇじゃないが
S:ある X X
U:わかる
S:だよね笑 T O
U:だよねー
S:だよねー X O
U:いやまねしないでください
S:ちょっと難しいですね T O
U:何いってるかわからないですね
S:語尾に付けますよねー X X
U:なにをだよ
S:アイコン T T
U:まじか
S:まじまじー O O


