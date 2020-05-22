import chainer
import chainer.functions as F
from chainer import Variable
import numpy as np
class Seq2SeqUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.seq2seq = kwargs.pop("models")
        super(Seq2SeqUpdater, self).__init__(*args, **kwargs)
    
    def loss(self,enc_words, dec_words, ARR):
        """
        :param enc_words: 発話文の単語を記録したリスト
        :param dec_words: 応答文の単語を記録したリスト
        :param model: Seq2Seqのインスタンス
        :param ARR: cuda.cupyかnumpyか
        :return: 計算した損失の合計
        """
        # バッチサイズを記録
        batch_size = len(enc_words[0])
        # model内に保存されている勾配をリセット
        self.seq2seq.reset()
        # 発話リスト内の単語を、chainerの型であるVariable型に変更
        # enc_words = [Variable(ARR.array(row, dtype='int32')) for row in enc_words]
        # エンコードの計算 ⑴
        self.seq2seq.encode(enc_words)
        # 損失の初期化
        loss = Variable(ARR.zeros((), dtype='float32'))
        # <eos>をデコーダーに読み込ませる (2)
        t = Variable(ARR.array([0 for _ in range(batch_size)], dtype='int32'))
        # デコーダーの計算
        for w in dec_words:
            # 1単語ずつデコードする (3)
            y = self.seq2seq.decode(t)
            # 正解単語をVariable型に変換
            t = Variable(ARR.array(w, dtype='int32'))
            # 正解単語と予測単語を照らし合わせて損失を計算 (4)
            loss += F.softmax_cross_entropy(y, t)
        return loss

    def update_core(self):
        seq2seq_optimizer = self.get_optimizer('seq2seq')
        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        xp = chainer.cuda.get_array_module(batch)
        enc_words, dec_words =xp.split(xp.array(batch), 2, axis=1)
        # enc_words = xp.array(batch).
        # dec_words = xp.array(batch).T[1]
        # print(enc_words)
        seq2seq_optimizer.update(self.loss, enc_words, dec_words, xp)
