import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_1(x,Y,y):
    plt.scatter(x,Y,c='r')
    plt.plot(x,y,c='blue')
    plt.show()


# 定义数据集
step = 1000
rate = 0.01

X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
X = np.vstack( [X,np.ones([1,len(X)])] ).T
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3]).reshape(17,1)

w = tf.Variable(tf.random_normal([2,1],stddev=1))

x = tf.placeholder(tf.float32,shape=(len(X),2))
y_ = tf.placeholder(tf.float32,shape=(len(Y),1))

y = tf.matmul(x,w)

# 损失函数
cross_entropy = tf.reduce_mean( tf.pow(y-y_,2) )
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(step+1):
        sess.run(train_step,
                 feed_dict={x:X,y_:Y})
        if i % 200 ==0:
            loss = sess.run(cross_entropy,
                            feed_dict={x:X,y_:Y})
            print("the {} time, loss: {}".format(i,loss))

    weight = sess.run(w)
    new_y = sess.run(y,feed_dict={x:X,y_:Y})

print(weight,np.shape(X[:,0]),np.shape(list(Y[:,0])))
print(np.shape(new_y[:,0]))
plot_1(X[:,0],Y[:,0],new_y[:,0])
