import tensorflow as tf

hello = tf.constant('Hello, Ada from Tensorflow')
session = tf.Session()
print(session.run(hello))
