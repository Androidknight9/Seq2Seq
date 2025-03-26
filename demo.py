import os.path
import random
import re
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.optim import AdamW
from tqdm import tqdm

# todo:0- 初始化全局变量
# 设备选择, 我们可以选择在cuda或者cpu上运行你的代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = './data/cmn.txt'

SOS_token = 0

EOS_token = 1

MAX_LENGTH = 38


def normalizeString(s: str):
    """
    字符串规范化函数
    :param s:传入的字符串 e.g. how are you?\n
    :return:格式化后的字符串 s e.g. how are you ?
    """

    # todo:1.1- 将句子中的所有字母转换为小写(中文不受影响)并去掉两端空白,主要是句子末尾的换行符
    s = s.lower().strip()
    # print('step1->', s)

    """
    re.sub(pattern,repl,string)
        替换字符串中所有匹配正则表达式模式的部分，返回替换后的新字符串。
    参数解释:
        - pattern:匹配模式
        - repl:替换的字符串或函数
        - string:原始字符串
        - count:替换的最大次数
        - flag:匹配模式表示 
            - 例如:re.IGNORECASE（忽略大小写）、re.MULTILINE（多行模式）等。
    """
    # todo:1.2- 通过正则表达式将所有标点符号替换成 空格+标点符号 形式 "!"-> " !" e.g
    s = re.sub(r"([.!?])", r" \1", s)
    # print('step2->', s)

    # todo:1.3- 通过正则表达式将除 a-z.!? 之外的符号替换为空格 " "
    s = re.sub(r"[^a-z\u4e00-\u9fa5 .!?。！？]+", r" ", s)
    # print('step3->', s)

    return s


def data_preprocessing(path=DATA_PATH):
    data_frame = pd.read_csv(path, sep='\t', names=['en', 'ch', 'info'], usecols=[0, 1])
    data_frame['en'] = data_frame['en'].apply(lambda x: normalizeString(x))
    data_frame['ch'] = data_frame['ch'].apply(lambda x: normalizeString(x))

    pair_list = data_frame.values.tolist()
    print(f'pari_list->', pair_list[:5])
    # todo:2.3.1- 构建英文单词-下标 词汇表 `english_word2index` 英文词汇表长度 `english_word_n`
    english_word2index = {'_SOS': SOS_token, '_EOS': EOS_token}
    # 第三个单词的下标从2开始
    english_word_n = 2

    # todo:2.3.2- 中文单词-下标 词汇表 `mandarin_word2index` 中文词汇表长度 `mandarin_word_n`
    mandarin_word2index = {'_SOS': SOS_token, '_EOS': EOS_token}
    # 第三个单词的下标从2开始
    mandarin_word_n = 2
    for pair in pair_list:
        for en in pair[0].split(' '):
            if en not in english_word2index:
                english_word2index[en] = english_word_n
                english_word_n += 1
        for ch in pair[1]:
            if ch not in mandarin_word2index:
                mandarin_word2index[ch] = mandarin_word_n
                mandarin_word_n += 1
    print('english_word_n->', english_word_n)
    print('mandarin_word_n->', mandarin_word_n)

    mandarin_index2word = {v: k for k, v in mandarin_word2index.items()}
    english_index2word = {v: k for k, v in english_word2index.items()}
    return pair_list, english_word2index, mandarin_word2index, english_index2word, mandarin_index2word, english_word_n, mandarin_word_n


# todo:3- 构建数据集类 PairsDataset
class PairsDataset(Dataset):
    # todo:3.1- `__init__`方法,用于初始化属性
    def __init__(self, pair_list, english_word2index, mandarin_word2index):
        self.pairs = pair_list
        # 样本 x
        self.english_word2index = english_word2index
        # 样本 y
        self.mandarin_word2index = mandarin_word2index

        # 样本条目数
        self.samples_len = len(self.pairs)

    # todo:3.2- `__len__方法`,返回数据集样本个数 tips:使用`len`方法时自动调用
    def __len__(self):
        return self.samples_len

    # todo:3.3- `__getitem__`方法,将句子数值化后返回对应张量 tips:下标取值时自动调用
    def __getitem__(self, index):
        # todo:3.1-  索引下标修正,防止取元素的时候出现越界
        index = min(max(0, index), self.samples_len - 1)
        # todo:3.2- 文本数值化
        # 获取每行数据=> self.pairs[index]=> ['英文句子','中文句子']
        ## 获取英文句子 x
        x = self.pairs[index][0]
        ## 获取中文句子 y
        y = self.pairs[index][1]
        ## 根据空格分割句子成词并根据词汇表转换为下标索引
        x = [self.english_word2index[word] for word in x.split(' ')]
        ## 每行句子都要添加一个结束标识符 _EOS
        x.append(EOS_token)
        # x.extend([SOS_token] * (ENG_MAX_LENGTH - len(x)))
        y = [self.mandarin_word2index[word] for word in y]
        ## 每行句子都要添加一个结束标识符 _EOS
        y.append(EOS_token)

        # todo:3.3- 文本张量化并返回
        tensor_x = torch.tensor(x, dtype=torch.long)
        tensor_y = torch.tensor(y, dtype=torch.long)
        return tensor_x, tensor_y


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=4, bidirectional=False, dropout=0.5):
        """
        搭建基于LSTM的编码器端的网络层
        :param input_size: 英文词汇表大小
        :param hidden_size: 隐藏层大小
        :param embedding_dim: 词向量维度=input_dim
        :param num_layers: 隐藏层层数
        :param bidirectional: 是否为双向网络
        :var self.num_direction: 网络方向数(0或1)
        """
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_direction = int(bidirectional) + 1

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            num_layers=num_layers)

    def forward(self, input, h0, c0):
        """
        定义网络层前向传播方法,生成一个初始语义张量c
        :param input: 二维张量[batch_size,seq_len]
        :param c0: 编码器端初始细胞状态
        :param h0: 编码器端初始隐藏状态
        :return:output,hn
        """
        # todo:5.2.1- 数据经过词嵌入层,升维
        # [batch_size,seq_len] => [batch_size,seq_len,embedding_dim]
        input = self.embedding(input.to(device=device))

        # todo:5.2.2- 数据经过 `GRU`层得到 `output` 和 `hn`
        # output:[batch_size,seq_len,hidden_size*num_direction]
        # hn:[num_layers * num_direction,batch_size,hidden_size]
        output, (hn, cn) = self.lstm(input, (h0, c0))
        return output, hn, cn

    def initParams(self):
        h0 = torch.zeros(self.num_direction * self.num_layers, 1, self.hidden_size, device=device)
        c0 = torch.zeros_like(h0, device=device)
        return h0, c0


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, num_layers=4, bidirectional=False, dropout=0.5):
        """"""
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_direction = int(bidirectional) + 1

        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=self.hidden_size * self.num_direction, out_features=self.input_size)

    def forward(self, input, h0, c0):
        """
        传统Seq2Seq的前向传播过程=>数据是一个一个喂给模型的,也就是 GRU=>丢给线性层产生一个分类分数=>softmax激活=>预测一个词=>作为下一个时间步的输入
        :param input:解码器的输入:(1,1) => (batch_size,1),一个词一个词的输入
        :param h0: 解码器的初始隐藏状态 = context vector
        :param c0: 解码器的初始细胞状态 = context vector
        :return:
        """
        input = self.embedding(input.to(device=device))

        # 进行 `relu`激活,将低维稠密张量 "稀疏化"=>防止过拟合
        # input = torch.relu(input)

        output, (hn, cn) = self.lstm(input, (h0, c0))

        output = self.fc(output[0])

        return output, hn, cn


def train():
    pair_list, english_word2index, mandarin_word2index, english_index2word, mandarin_index2word, english_word_n, mandarin_word_n = data_preprocessing(
        path=DATA_PATH)
    dataset = PairsDataset(pair_list, english_word2index, mandarin_word2index)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True
    )
    encoder = Encoder(input_size=english_word_n, embedding_dim=256, hidden_size=256, num_layers=4, bidirectional=True,
                      dropout=0.5).to(device=device)
    decoder = Decoder(input_size=mandarin_word_n, embedding_dim=256, hidden_size=256, num_layers=4, bidirectional=True,
                      dropout=0.5).to(device=device)
    if os.path.exists('./models/encoder.pth') and os.path.exists('./models/decoder.pth'):
        print('加载模型')
        encoder.load_state_dict(torch.load('./models/encoder.pth', map_location=lambda storage, loc: storage),strict=True)
        decoder.load_state_dict(torch.load('./models/decoder.pth', map_location=lambda storage, loc: storage),strict=True)
    optim = [AdamW(encoder.parameters(), lr=1e-3), AdamW(decoder.parameters(), lr=1e-3)]
    criterion = nn.CrossEntropyLoss()
    MAX_EPOCH = 20
    teacher_forcing_ratio = 0.8
    best_loss = float(18.0)  # 第一次迭代时为 'inf'
    for epoch in range(MAX_EPOCH):
        print(f'EPOCH:[{epoch}/{MAX_EPOCH}]')
        batch_loss = 0.0
        batch_acc_num = 0
        iter_token = 0
        iter_num = 0
        start_time = time.time()
        for x, y in tqdm(data_loader):
            encoder.train()
            decoder.train()
            y = y.to(device=device)

            output_encoder, hn, cn = encoder(x, *encoder.initParams())
            input_y = torch.tensor([[SOS_token]], device=device)
            loss = 0.0
            y_len = y.shape[1]
            iter_token += y_len
            use_teacher_forcing = (True if random.random() < teacher_forcing_ratio else False)
            if use_teacher_forcing:
                for idx in range(y_len):
                    output_y, hn, cn = decoder(input_y, hn, cn)
                    target_y = y[0][idx].view(1)
                    loss += criterion(output_y, target_y)
                    batch_loss += loss.item() / y_len
                    batch_acc_num += 1 if torch.argmax(output_y, -1).item() == target_y.item() else 0
                    input_y = y[0][idx].view(1, -1)
            else:
                for idx in range(y_len):
                    output_y, hn, cn = decoder(input_y, hn, cn)
                    target_y = y[0][idx].view(1, )
                    loss += criterion(output_y, target_y)
                    batch_loss += loss.item() / y_len
                    batch_acc_num += 1 if torch.argmax(output_y, -1).item() == target_y.item() else 0
                    topv, topi = output_y.topk(1)
                    if topi.item() == EOS_token:
                        break
                    input_y = topi.detach()
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            optim[0].step()
            optim[1].step()
            iter_num += 1
            if iter_num % 1000 == 0:
                print(
                    f'time:{time.time() - start_time}, loss:{batch_loss / iter_num}, acc:{batch_acc_num / iter_token}')
        if best_loss > batch_loss:
            best_loss = batch_loss
            torch.save(encoder.state_dict(), f'./models/encoder.pth')
            torch.save(decoder.state_dict(), f'./models/decoder.pth')
            print('improved')


def translate():
    (pair_list, english_word2index, mandarin_word2index, english_index2word, mandarin_index2word, english_word_n,
     mandarin_word_n) = data_preprocessing(path=DATA_PATH)

    input_size = english_word_n
    hidden_size = 256
    encoder = Encoder(input_size=input_size, embedding_dim=256, hidden_size=hidden_size,
                      num_layers=4, bidirectional=True).to(device=device)

    encoder.load_state_dict(torch.load(f'./models/encoder.pth', map_location=lambda storage, loc: storage),
                            strict=True)
    input_size = mandarin_word_n
    hidden_size = 256
    decoder = Decoder(input_size=input_size, embedding_dim=256, hidden_size=hidden_size, num_layers=4,
                      bidirectional=True).to(device=device)
    decoder.load_state_dict(torch.load(f'./models/decoder.pth', map_location=lambda storage, loc: storage),
                            strict=True)
    # 创建预测数据对象
    pred_pairs = [['How old are you?', '你多大了?'],
                  ['This little baby tore up a 10 dollar bill.', '这个小婴儿撕毁了一张10美元的钞票。'],
                  ['Hi','你好'],
                  ['Stop!','住手！']]
    for pair in pred_pairs:
        x = pair[0]
        y = pair[1]
        tmp_list = list()
        x = normalizeString(x)
        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        x = torch.tensor(x, device=device).view(1, -1)

        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            output_encoder, hn, cn = encoder(x, *encoder.initParams())
            input_y = torch.tensor([[SOS_token]], device=device)
            for idx in range(MAX_LENGTH):
                output, hn, cn = decoder(input_y, hn, cn)
                topv, topi = output.topk(1)
                if topi.item() == EOS_token:
                    break
                else:
                    tmp_list.append(mandarin_index2word[topi.item()])
                input_y = topi.detach()
            # 将列表转换为字符串并打印
            trans = ' '.join(tmp_list)
            print('\n')
            print('>', x)
            print('=', y)
            print('<', trans)


if __name__ == '__main__':
    train()
    translate()
