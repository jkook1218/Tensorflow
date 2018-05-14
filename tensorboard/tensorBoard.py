import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter = ',', unpack = True, dtype = 'float32')

x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
print(x_data)
print(y_data)

global_step = tf.Variable(0, trainable = False, name = 'global_step') # 학습 횟수를 저장하는 Varibale을 생성!

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.name_scope('layer1'):
    W1 = tf.Variable( tf.random_uniform([2,10], -1. , 1. ))
    L1 = tf.nn.relu(tf.matmul(X, W1))

    tf.summary.histogram("Weights", W1)

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))
    y_hat = tf.matmul(L2, W3) #다음단계에 softmax가 포함되어있으므로 그냥 곱하기만 한다.

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels= Y, logits = y_hat))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.01)
    train_op = optimizer.minimize(cost, global_step = global_step)

    tf.summary.scalar('cost', cost)                                             #저장하고싶은 상수있는곳에 넣기

sess = tf.Session()
saver = tf.train.Saver(tf.global_variables())                                   # saver 암기, 전역변수를 저장하는것!
feed_dict= {X: x_data, Y: y_data}

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()                                                 # merged 라는 변수에 tf.summary로 저장한것 모두 통합
writer = tf.summary.FileWriter('./logs', sess.graph)                             # logs 폴더에 sess.graph와 tensor값들을 저장한다.

for step in range(10):
    sess.run(train_op, feed_dict= feed_dict)

    print('Step: %d' % sess.run(global_step), 'Cost: %d' % sess.run(cost, feed_dict= feed_dict))

    summary = sess.run(merged, feed_dict = feed_dict)
    writer.add_summary(summary, global_step = sess.run(global_step))

saver.save(sess, './model/dnn.ckpt', global_step = global_step)

prediction = tf.argmax(y_hat, axis =  1)
target = tf.argmax(Y, axis = 1)
print('예측값: ', sess.run(prediction, feed_dict = {X: x_data}) )
print('실제값: ', sess.run(target, feed_dict= {Y : y_data }))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도: %.2f' % sess.run(accuracy *100, feed_dict= feed_dict))
