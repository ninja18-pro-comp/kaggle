import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class LSTM_Encoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 中間層のサイズ
        """
        super(LSTM_Encoder, self).__init__(
            # 単語を単語ベクトルに変換する層
            xe = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを隠れ層の4倍のサイズのベクトルに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            # 出力された中間層を4倍のサイズに変換するための層
            hh = L.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        """
        Encoderの動作
        :param x: one-hotなベクトル
        :param c: 内部メモリ
        :param h: 隠れ層
        :return: 次の内部メモリ、次の隠れ層
        """
        print(x.shape)
        e = x
        e = self.xe(e)
        e = F.tanh(e)
        # 前の内部メモリの値と単語ベクトルの4倍サイズ、中間層の4倍サイズを足し合わせて入力
        return F.lstm(c, self.eh(e) + self.hh(h))
    
class LSTM_Decoder(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        """
        クラスの初期化
        :param vocab_size: 使われる単語の種類数（語彙数）
        :param embed_size: 単語をベクトル表現した際のサイズ
        :param hidden_size: 中間ベクトルのサイズ
        """
        super(LSTM_Decoder, self).__init__(
            # 入力された単語を単語ベクトルに変換する層
            ye = L.EmbedID(vocab_size, embed_size, ignore_label=-1),
            # 単語ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            eh = L.Linear(embed_size, 4 * hidden_size),
            # 中間ベクトルを中間ベクトルの4倍のサイズのベクトルに変換する層
            hh = L.Linear(hidden_size, 4 * hidden_size),
            # 出力されたベクトルを単語ベクトルのサイズに変換する層
            he = L.Linear(hidden_size, embed_size),
            # 単語ベクトルを語彙サイズのベクトル（one-hotなベクトル）に変換する層
            ey = L.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        """
        :param y: one-hotなベクトル
        :param c: 内部メモリ
        :param h: 中間ベクトル
        :return: 予測単語、次の内部メモリ、次の中間ベクトル
        """
        # 入力された単語を単語ベクトルに変換し、tanhにかける
        e = F.tanh(self.ye(y))
        # 内部メモリ、単語ベクトルの4倍+中間ベクトルの4倍をLSTMにかける
        c, h = F.lstm(c, self.eh(e) + self.hh(h))
        # 出力された中間ベクトルを単語ベクトルに、単語ベクトルを語彙サイズの出力ベクトルに変換
        t = self.ey(F.tanh(self.he(h)))
        return t, c, h

class Seq2Seq(chainer.Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
        """
        Seq2Seqの初期化
        :param vocab_size: 語彙サイズ
        :param embed_size: 単語ベクトルのサイズ
        :param hidden_size: 中間ベクトルのサイズ
        :param batch_size: ミニバッチのサイズ
        :param flag_gpu: GPUを使うかどうか
        """
        super(Seq2Seq, self).__init__(
            encoder = LSTM_Encoder(vocab_size, embed_size, hidden_size),
            decoder = LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # GPUで計算する場合はcupyをCPUで計算する場合はnumpyを使う
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def encode(self, words):
        """
        Encoderを計算する部分
        :param words: 単語が記録されたリスト
        :return:
        """
        # 内部メモリ、中間ベクトルの初期化
        c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        words = self.ARR.array(words.tolist())
        print(words.shape)
        # exit()
        # エンコーダーに単語を順番に読み込ませる
        for i in range(words.shape[2]):
            
        for w in words[:,0,:]:
            print(w.shape)
            c, h = self.encoder(w, c, h)

        # 計算した中間ベクトルをデコーダーに引き継ぐためにインスタンス変数にする
        self.h = h
        # 内部メモリは引き継がないので、初期化
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

    def decode(self, w):
        """
        デコーダーを計算する部分
        :param w: 単語
        :return: 単語数サイズのベクトルを出力する
        """
        t, self.c, self.h = self.decoder(w, self.c, self.h)
        return t

    def reset(self):
        """
        中間ベクトル、内部メモリ、勾配の初期化
        :return:
        """
        self.h = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size, self.hidden_size), dtype='float32'))

        self.zerograds()

