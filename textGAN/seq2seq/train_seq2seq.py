import argparse
import os
import chainer
from seq2seq  import Seq2Seq
from updater import Seq2SeqUpdater
from chainer import training
from chainer.training import extensions
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# random.seed(0)
# np.random.seed(0)

def load_dataset(word2id,N=20000):
    def generate_number():
        number = [random.choice(list("0123456789")) for _ in range(random.randint(1, 3))] 
        # a <= N <= b random.randint(a, b)
        return int("".join(number))
    
    def padding(string, training=True):
        string = "{:*<7s}".format(string) if training else "{:*<6s}".format(string)
        return string.replace("*", "<pad>")
    
    def transform(string, seq_len=7):
        tmp = []
        for i, c in enumerate(string):
            try:
                tmp.append(word2id[c])
            except:
                tmp += [word2id["<pad>"]] * (seq_len - i)
                break
        return tmp

    data = []
    target = []    
    for _ in range(N):
        x = generate_number()
        y = generate_number()
        z = x + y
        left = padding(str(x) + "+" + str(y))
        right = padding(str(z), training=False)
        data.append(transform(left))
        right = transform(right, seq_len=6)
        right = [12] + right[:5]
        right[right.index(10)] = 12
        target.append(right)
        
    return data, target

# def train2batch(data, target, batch_size=100):
#     input_batch = []
#     output_batch = []
#     data, target = shuffle(data, target)
    
#     for i in range(0, len(data), batch_size):
#         input_tmp = []
#         output_tmp = []
#         for j in range(i, i+batch_size):
#             input_tmp.append(data[j])
#             output_tmp.append(target[j])
#         input_batch.append(input_tmp)
#         output_batch.append(output_tmp)
#     return input_batch, output_batch

def main():
    parser = argparse.ArgumentParser(description='Chainer: Seq2Seq')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    args = parser.parse_args()

    word2id = {str(i): i for i in range(10)}
    word2id.update({"<pad>": 10, "+": 11, "<eos>": 12})
    id2word = {v: k for k, v in word2id.items()}

    vocab_size = len(word2id)
    embed_size = 16
    hidden_size = 128
    batch_size = 64
    epoch = 100
    xp = chainer.cuda.cupy

    data, target = load_dataset(word2id)
    # data = xp.array(data,dtype=xp.int32)
    # target = xp.array(target,dtype=xp.int32)
    train_x, test_x, train_t, test_t = train_test_split(data, target, test_size=0.1)
    train = list(zip(train_x,train_t))
    # train = chainer.datasets.TupleDataset(train_x,train_t)
    # print(train)
    # exit()
    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    # Set up a neural network to train
    seq2seq = Seq2Seq(vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True)
 
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        seq2seq.to_gpu()
 
    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer
 
    opt_seq2seq = make_optimizer(seq2seq)

    # Set up a trainer
    updater = Seq2SeqUpdater(models=seq2seq, iterator=train_iter, optimizer={'seq2seq':opt_seq2seq}, device=args.gpu)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=args.out)
 
    epoch_interval = (1, 'epoch')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'seq2seq/loss']), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(('seq2seq/loss', 'dis/loss')))

    # if args.resume:
    #     # Resume from a snapshot
    #     chainer.serializers.load_npz(args.resume, trainer)
 
    # Run the training
    trainer.run()
 
 
if __name__ == '__main__':
    main()