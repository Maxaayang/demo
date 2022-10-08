import pickle

seq_len = pickle.load(open('../seq_len', 'rb'))

size = seq_len.shape[0]
for i in range(size):
    if (i > 0):
        seq_len[i] += seq_len[i - 1]

pickle.dump(seq_len, open('../seq_len', 'wb'))