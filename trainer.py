
import numpy as np
from tensorflow import keras
from keras import layers
from config import *
from tokenizer.name_tokenizer import NameTokenizer


def generate_data(tokenizer: NameTokenizer):

    total = len(tokenizer.names)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)

        sequences = []
        next_chars = []

        for name in tokenizer.names[start:end]:
            s = name + (MAX_LEN - len(name))*'0'
            sequences.append(s)
            next_chars.append('\n')
            for it, j in enumerate(name):
                if (it >= len(name)-1):
                    continue
                s = name[:-1-it]+(MAX_LEN - len(name[:-1-it]))*'0'
                sequences.append(s)
                next_chars.append(name[-1-it])

            # print(sequences[:10])
            # print(next_chars[:10])
        x_train = np.zeros((len(sequences), MAX_LEN, tokenizer.vocab))
        y_train = np.zeros((len(next_chars), tokenizer.vocab))
        for idx, seq in enumerate(sequences):
            for t, char in enumerate(seq):
                x_train[idx, t, tokenizer.token_id_dict[char]] = 1
        for idx, char in enumerate(next_chars):
            y_train[idx, tokenizer.token_id_dict[char]] = 1
        yield x_train, y_train
        del sequences,next_chars,x_train,y_train


def train():

    tokenizer = NameTokenizer(path=PATH_NAME, required_gender=GENDER)
    steps = len(tokenizer.names)/BATCH_SIZE
    # # 构建模型
    model = keras.Sequential([
        keras.Input(shape=(MAX_LEN, tokenizer.vocab)),
        # 第一个LSTM层，返回序列作为下一层的输入
        layers.LSTM(128, dropout=DROP_OUT, return_sequences=False),
        layers.Dense(tokenizer.vocab, activation='softmax')
    ])
    model.summary()
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.fit_generator(generate_data(tokenizer),
                        steps_per_epoch=steps, epochs=EPOCHS)

    model.save()
