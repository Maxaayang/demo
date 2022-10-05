import tensorflow as tf

def f(x:tf.data.Dataset):
    a=[i for i in x]
    return a

a = tf.data.Dataset.range(9)
b = a.window(3)
c = b.map(f)

print(list(c.as_numpy_iterator()))
