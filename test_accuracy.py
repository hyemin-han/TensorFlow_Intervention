import tensorflow as tf
import test_class

import numpy as np
import time

errors = []
nowerror = [0.0]
times = []
nowtime = [0.0]

## weight
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.01)
	return tf.Variable(initial)

## bias
def bias_variable(shape):
	initial = tf.constant(0.01, shape=shape)
	return tf.Variable(initial)

## convolution function
def conv2d (x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1],
		padding = 'SAME')

## pooling 2x2
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1,2,1,1],
		strides=[1,2,1,1], padding = 'SAME')

## read test data
mnist = test_class.read_from_csv()

## create TensorFlow session
sess = tf.InteractiveSession()

## initiation
output = np.zeros((len(mnist.y),2))

## change the format (to two binary variables)
for i in range(len(mnist.y)):
	if mnist.y[i] == 0:
		output[i][0] = 1
		output[i][1] = 0
	else:
		output[i][0] = 0
		output[i][1] = 1

lockboxx =[]
lockboxy =[]

now = 0

for j in range(1):
	f = open('randompick_test.txt','a')
	now = now + 1

	## creating x and y placeholders
	x = tf.placeholder(tf.float32, [None, 8])
	y_ = tf.placeholder(tf.float32, [None, 2])

	## 1st convolutional layer
	W_conv1 = weight_variable([2,1,1,32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1,8,1,1])

	## apply ReLU function
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	## 2nd convolutional layer
	W_conv2 = weight_variable([2,1,32,64])
	b_conv2 = bias_variable([64])

	## apply ReLU function
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	## 1st dense layer
	W_fc1 = weight_variable([2*1*64,1024])
	b_fc1 = bias_variable([1024])

	## apply ReLu function
	h_pool2_flat = tf.reshape(h_pool2, [-1, 2*1*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	## 2nd dense layer
	W_fc2 = weight_variable([1024,2])
	b_fc2 = bias_variable([2])

	## calculate final outcome
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+ b_fc2)

	## automatically attempt to minimize the cross entropy for feedback
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv),reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	## calculate prediction accuracy at this step
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	## before get started, leave 30 cases
	toleave = test_class.shuffle_order(len(mnist.x),32)

	## exclude "to leave" datasets (leave lockbox dataset)
	tolearncount = 0

	lockboxcount = 0
	for i in range(len(mnist.x)):
		flag = 0
		for j in range(32):
			if i == toleave[j]:
				flag = 1
				if lockboxcount > 0:
					lockboxx = np.vstack([lockboxx,mnist.x[i]])
					lockboxy = np.vstack([lockboxy,output[i]])
					lockboxcount = lockboxcount + 1
				else:
					lockboxx = mnist.x[i]
					lockboxy = output[i]
					lockboxcount = lockboxcount + 1	
		if flag == 0:
			if tolearncount > 0:
				tolearndata = np.vstack([tolearndata,mnist.x[i]])
				tolearny = np.vstack([tolearny,output[i]])
				tolearncount = tolearncount + 1		
			else:
				tolearndata = mnist.x[i]
				tolearny = output[i]
				tolearncount = tolearncount + 1


	sess.run(tf.initialize_all_variables())

	start_time = time.time()

	## 1000 iterative learning. "1000" can be modified per user's convenience.
	for i in range(1000):

		## Shuffle it! -> extact learning data (70%)
		arr = np.arange(len(tolearndata))

		## shuffle it
		np.random.shuffle(arr)

		## only return the first =count= indices
		outputs = arr[0:len(tolearndata)]
	
		batch_xs=tolearndata[outputs]
		batch_ys = tolearny[outputs]

		## Only 90% of data (except for the lockbox cases) is used for learning process. 
		## conduct learning for 1000 steps
		train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})


	batch_xs = mnist.x

	## now, test the whole data.
	y = sess.run(y_conv,feed_dict={x: mnist.x, keep_prob: 0.5})

	## Print accuracy data (For both the whole data and lockbox data)
	print ("%d test accuracy %g" % (now, accuracy.eval(feed_dict={x: mnist.x, y_:output, keep_prob: 1.0})))
	print ("lockbox test accuracy %g" % accuracy.eval(feed_dict={x: lockboxx, y_:lockboxy, keep_prob: 1.0}))
	## predicted output data is stored in a text file
	nowoutput = "%d %g %g\r\n" % (now, accuracy.eval(feed_dict={x: mnist.x, y_:output, keep_prob: 1.0}), accuracy.eval(feed_dict={x: lockboxx, y_:lockboxy, keep_prob: 1.0}))
	f.write(nowoutput)
	f.close

saver = tf.train.Saver()
saver.save(sess,'purpose-model')


