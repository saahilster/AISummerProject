import os
os.environ['TF_CPP_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
print(tf.__version__)

#Initializing Tensors
#x = tf.constant(1)

# manual way of initializing
#y = tf.constant([[1,2,3], [4,5,6]])


#/ faster way
#x = tf.ones((4, 4))
#y = tf.zeros((2,3))
#z = tf.eye(3)

#Operations w/ tensors
#x = tf.constant([2,1,3])
#y = tf.constant([3,4,2])

#z = x + y

    #division
#z = x / y

    #matrix mult
x = tf.random.normal((2,1,1))
y = tf.random.normal((2,1,1))
z = x @ y
print(z)


#Indexing

#Reshaping