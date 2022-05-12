import numpy as np
from tensorflow import keras
from config import *

from tokenizer.name_tokenizer import NameTokenizer
def sample(preds, temperature=1):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generateNames(prefix,size):

    tokenizer = NameTokenizer(path=PATH_DATA, required_gender=GENDER)
    model = keras.models.load_model(PATH_MODEL)
    preds = set()
    
    
    tmp_generated = prefix

    for char in tmp_generated:
        if char not in tokenizer.token_id_dict:
            print("字典中没有这个字")
            return

    sequence = ('{0:0<' + str(MAX_LEN) + '}').format(prefix).lower()
    
    while len(preds) < size:

        x_pre = np.zeros((1, MAX_LEN, tokenizer.vocab))
        for t, char in enumerate(sequence):
            x_pre[0, t, tokenizer.token_id_dict[char]] = 1
        output = model.predict(x_pre, verbose=0)[0]
        index = sample(output)
        char = tokenizer.id_token_dict[index]

        if(char == '0' or char == '\n'):
            preds.add(tmp_generated)
            tmp_generated = prefix
            sequence = ('{0:0<' + str(MAX_LEN) + '}').format(prefix).lower()
        else:
            tmp_generated += char
            sequence = (
                '{0:0<' + str(MAX_LEN) + '}').format(tmp_generated).lower()
        print(sequence)

        if(len(sequence) > MAX_LEN):
            tmp_generated = prefix
            sequence = ('{0:0<' + str(MAX_LEN) + '}').format(prefix).lower()
    return preds