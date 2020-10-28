# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

# ## -------------one-------------###
# creat data
# x_data= np.random.rand(100).astype(np.float)
# y_data= x_data*0.1+ 0.3
#
# # creat structure
# weight= tf.Variable(tf.random_uniform([1],-1,1))
# biases= tf.Variable(tf.zeros([1]))
#
# y =weight* x_data+ biases
#
# loss= tf.reduce_mean(tf.square(y- y_data))
# optimizer= tf.train.GradientDescentOptimizer(0.5)
# train= optimizer.minimize(loss)
#
# init= tf.initialize_all_variables()
# # creat struct end
#
# sess= tf.Session()
# sess.run(init)
#
# for t in range(202):
#     sess.run(train)
#     if t %50 ==0:
#         print(t,sess.run(weight),sess.run(biases))

# # ##-------------two----------------###
#
# mat1= tf.constant([[3,3]])
# mat2= tf.constant([[2],
#                    [2]])
# product= tf.matmul(mat1,mat2)
#
# # # method 1
# # sess= tf.Session()
# # result= sess.run(product)
# # print(result)
# # sess.close()
#
# # method 2
# with tf.Session() as sess:
#     relult2= sess.run(product)
#     print(relult2)

# --------------------three-----------------------#

# state= tf.Variable(0,name="My")
# # print(state.name)
# one= tf.constant(1)
#
# new_value= tf.add(state,one)
# update= tf.assign(state,new_value)
#
# init= tf.initialize_all_variables()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(5):
#         sess.run(update)
#         print(sess.run(new_value),sess.run(state))

# ------------------four-------------------------#

# input1= tf.placeholder(tf.float32)
# input2= tf.placeholder(tf.float32)
#
# oupput= tf.multiply(input1,input2)
#
# with tf.Session() as sess:
#     print(sess.run(oupput,feed_dict={input1:[7.],
#                                      input2:[2.]}))

# -------------------------five------------------------------#

# def add_layer(inputs,in_size,out_size,activation_fuction=None):
#
#     Weights= tf.Variable(tf.random_normal([in_size,out_size]),name="Weights")
#     biases= tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
#     Wx_plus_b= tf.matmul(inputs,Weights )+ biases
#     if activation_fuction is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_fuction(Wx_plus_b)
#     return outputs
#
# x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]
# noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)
# y_data = np.square(x_data)- 0.5 + noise

# xs = tf.placeholder(tf.float32,[None,1],name="x_input")
# ys = tf.placeholder(tf.float32,[None,1],name="y_input")
# l1 = add_layer(xs,1,10,activation_fuction=tf.nn.relu)

# prediction= add_layer(l1,10,1,activation_fuction=None)
#
# loss= tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
#                     reduction_indices=[1]))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
# init = tf.global_variables_initializer()
# sess =tf.Session()
#
# sess.run(init)
#
# #-----画图部分---加上最后两行---
# # fig = plt.figure()
# # ax = fig.add_subplot(1,1,1)
# # ax.scatter(x_data,y_data)
#
# for i in range(1002):
#     sess.run(train_step, feed_dict={xs:x_data,ys:y_data})
#     if i % 50 ==0:
#         print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#         # try:
#         #     ax.lines.remove(lines[0])
#         # except Exception:
#         #     pass
#         prediction_value = sess.run(prediction,feed_dict={xs:x_data})
#         # lines = ax.plot(x_data,prediction_value,'red',lw=4)
#         # plt.pause(0.1)
#
# # ----------------------------six---------------------------------
# mnist = input_data.read_data_sets('MINIS_data', one_hot='true')
#
#
# def add_layer(inputs, in_size, out_size, activation_fuction=None):
#     Weights = tf.Variable(tf.random_normal([in_size, out_size]))
#     biases = tf.Variable(tf.zeros([1, out_size]))
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if activation_fuction is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = activation_fuction(Wx_plus_b)
#     return outputs
#
#
# def compute_accuracy(v_xs, v_ys):
#     global prediction
#     y_pre = sess.run(prediction, feed_dict={xs: v_xs,keep_drop:0.5})
#     correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys,keep_drop:0.5})
#     return result
#
#
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
#
# def bias_Variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
#
# def conv2d(x, W):  # strides[1,x_movement,y_movement,1]
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding="SAME")  # strides[1,x_movement,y_movement,1]
#
#
# xs = tf.placeholder(tf.float32, [None, 784])
# ys = tf.placeholder(tf.float32, [None, 10])
# keep_drop=tf.placeholder(tf.float32)
# x_image = tf.reshape(xs, [-1, 28, 28, 1])
#
# # conv l layer
# W_conv1 = weight_variable([5, 5, 1, 32])  # patch 5x5 in size 1, out size 32
# b_conv1 = bias_Variable([32])
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28x28x32
# h_pool1 = max_pool_2x2(h_conv1)  # output size 14x14x32
#
# # conv 2 layer
# W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5 in size 32, out size 64
# b_conv2 = bias_Variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14x14x64
# h_pool2 = max_pool_2x2(h_conv2)  # output size 7x7x64
#
# # func1 layer
# h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# W_fc1 = weight_variable([7 * 7 * 64, 1024])
# b_fc1 = bias_Variable([1024])
# # [n,7,7,64] -> [n,7*7*64]
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_drop)
#
# # func2 layer
# W_fc2 = weight_variable([1024, 10])
# b_fc2 = bias_Variable([10])
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
#
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(1002):
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#         sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_drop:0.5})
#         if i % 100 == 0:
#             print("第%d次：" % i, end="")
#             print(compute_accuracy(mnist.test.images, mnist.test.labels))
# -----------------------seven---------------------------
# 保存数据
# W = tf.Variable([[1,2,3],[4,5,6]],dtype=tf.float32,name='weight')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,save_path='me_one/save_net.ckpt')
#     print("Save path:",save_path)

# 将保存到数据打印
# W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weight')
# b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     saver.restore(sess,'me_one/save_net.ckpt')
#     print('Weight',sess.run(W),'\n','biases',sess.run(b))
# ----------------------------matplotlib----------------------------------
# x = np.linspace(-5,5,50)
# y2 = 2*x +4
# y1 = x**2
#
# # plt.figure(num=5,figsize=(5,5))
# plt.figure()
# # 设置坐标轴的长度
# # plt.xlim((-10,10))
# # plt.ylim((-10,10))
# # 设置坐标轴的名称
# plt.xlabel('xxx')
# plt.ylabel("yyy")
# # 设置某坐标点的名称
# # plt.xticks([4,6,9],
# #            ['bottom','middle','above'])
# # plt.yticks([0,2,4],
# #            [r'$\beta$',r'$magic$',r'$labour$'])

# # # 画线
# l1, = plt.plot(x,y2)
# l2, = plt.plot(x,y1,color = '#FFAAFF',linewidth = 2 ,linestyle ='--')
# # 在图上表示线的名称
# plt.legend(handles = [l1,l2],labels=['aaa','bbb'],loc=('best'))

# # # 设置坐标轴样式
# # ax = plt.gca()
# # #   轴的设置
# # ax.spines['right'].set_color('none')
# # ax.spines['top'].set_color('none')
# # ax.spines['left'].set_position(('data',0))
# # ax.spines['bottom'].set_position(('data',0))
# # #   轴上数字位置设置
# # # ax.xaxis.set_ticks_position(('data',0))
# # # ax.yaxis.set_ticks_position(('data',0))

# -------2 线和注释
# x=np.linspace(-10,10,20)
# y = 2* x
# x0 = 3
# y0 = 2*x0
#
# plt.figure()
# ax = plt.gca()
# #   轴的设置
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.spines['left'].set_position(('data',0))
# ax.spines['bottom'].set_position(('data',0))
# # 注释
# plt.annotate(r'$ 2*x=%s $'% y0,xy=(x0,y0),xycoords='data',xytext=(+20,-30),textcoords='offset points',
#              fontsize = 16,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.3'))
# plt.plot([x0,x0],[y0,0],linestyle='--')
# plt.text(-5,10,'你好呀')
#
# plt.plot(x,y,linewidth=33,zorder=1)  # zorder 和下面的循环搭配，为了让轴上的数字有框
# plt.scatter(x0,y0,s=50,color='black')
#
# for label in ax.get_xticklabels() + ax.get_yticklabels():
#     label.set_fontsize(12)
#     label.set_bbox(dict(facecolor= 'white',edgecolor='None',alpha=0.7))
#
# plt.show()
# -------------------3 beautiful 点集

# x = np.random.normal(0,1,1024)
# y = np.random.normal(0,1,1024)
# T = np.arctan2(x,y)
#
# plt.scatter(x,y,s=77,c=T,alpha=0.5)
# plt.xlim(-1.5,1.5)
# plt.ylim(-1.5,1.5)
# plt.xticks(())
# plt.yticks(())
#
# plt.show()

# ------------------------ 4 柱状图
# n = 12
# X = np.arange(n)
# Y1 = np.random.uniform(0,1,n)
# Y2 = np.random.uniform(0,1,n)
#
# plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
# plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')
# for x,y in zip(X,Y1):
#     plt.text(x,y+0.02,s='%.2f'%y,ha='center',va='bottom',color='#9999ff')
# for x,y in zip(X,Y2):
#     plt.text(x,-y-0.1,s='%.2f'%y,ha='center',va='bottom',color='#ff9999')
# plt.show()

# ----------------------------- 等高线
# def f(x,y):
#     return (1- x/2 + x**5 + y**3)*np.exp(-x**2 - y**2)
#
# n = 265
# x = np.linspace(-3,3,n)
# y = np.linspace(-3,3,n)
# X,Y = np.meshgrid(x,y)  # 生成网格
#
# plt.contourf(X,Y,f(X,Y),10,alpha = 0.65 ,cmap=plt.hot())
# C = plt.contour(X, Y, f(X, Y), 10, linewidths=0.5)
# plt.clabel(C, inline=True,colors='blue',fontsize=10)
# plt.scatter(0,0)
# plt.show()

# ------------------------------image

# a =np.array(np.linspace(0,1,9)).reshape(3,3)
# plt.imshow(a,interpolation='nearest')
# plt.show()

# --------------------------------------3D
# fig = plt.figure()
# ax = Axes3D(fig)
#
# X = np.linspace(-4,4,50)
# Y = np.linspace(-4,4,50)
# X,Y = np.meshgrid(X,Y)
# Z = np.cos(np.sqrt(X**2 + Y**2))
#
# ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap='rainbow')
# ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')
# ax.set_zlim(-2,2)
# plt.show()

# -------------------------------------- 分区展示
# -----------one
# plt.figure()
#
# plt.subplot(2,1,1)  # 在二行一列的表格上放第一个图
# plt.plot([0,1],[0,1])
#
# plt.subplot(2,3,4)  #在二行三列的表格上放第四个图
# plt.plot([0,1],[0,1])
# plt.subplot(2,3,5)
# plt.plot([0,1],[0,1])
# plt.subplot(2,3,6)
# plt.plot([0,1],[0,1])

# -----------two
# f,((ax11,ax12),(ax21,ax22))=plt.subplots(2,2)
# ax11.scatter([3,2],[1,1])
#
# plt.tight_layout()
# plt.show()

# ------------------------------------ 图中图
# x=np.linspace(1,10,10)
# y = np.sin(x)
# plt.figure()
#
# plt.axes([0.1,0.1,0.8,0.8]).plot(x,y)   # [left,bottom,width,height]
# plt.axes([0.6,0.1,0.25,0.25]).plot(x,-y)
#
# plt.show()

# ----------------------- 图的镜像
# x = np.linspace(1,20,20)
# y = np.sqrt(x)
#
# fig = plt.figure()
# ax1 = fig.add_subplot()
# ax1.plot(x,y)
#
# ax2 = ax1.twinx()
# ax2.plot(x,-y)
#
# plt.show()
#
# ---------------------------------动画
# fig, ax = plt.subplots()
#
# x = np.arange(0, 2*np.pi, 0.01)
# line, = ax.plot(x, np.sin(x))
#
# def init():
#     line.set_ydata(len(x))
#     return line,
#
# def animate(i):
#     line.set_ydata(np.sin(x + i /50))  # update the data.
#     return line,
#
# ani = animation.FuncAnimation(
#     fig, animate, init_func=init, interval=1, blit=True)
