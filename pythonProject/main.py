import os
os.environ['TF_CPP_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print(tf.__version__)

#Initializing Tensors
x = tf.constant(1)
print(x)

#Operations w/ tensors

#Indexing

#Reshaping