import tensorflow as tf
import numpy as np
import shutil
import os

out_path = 'models'
out_fname = 'graph-mlp'
train_file = out_path + '/'+out_fname + '.ckpt'


x_i = []
y_i = []

#import training data
in_path = "nn.mkm";
with open(in_path, 'r') as handle:
    for line in handle:
        if not line.strip():
            continue  # This skips blank lines
        tuplet = line.split(', ')
        _x = map(int, tuplet[0].split(' '))  
        _y = map(int, tuplet[1][1:].rstrip().split(' '))  
        
        #print _x, " - ", _y
        for i in xrange(0, len(_y), 2):
            y_i.append(_y[i:i+2])
            x_i.append(_x)


#in case binary representation ins needed
# max_input_size = 5
# beat_resolution = 4
# pitch_resolution = 7
# output_size = beat_resolution + pitch_resolution
# print '{0:08b}'.format(6)[6:7]
n_sample = len(x_i)
max_input_size = 5
output_size = 2

for i in range(len(x_i)):
    for j in range(max_input_size - len(x_i[i])):
        x_i[i].insert(0, 0)

x_t = [np.array(x)/128. for x in x_i]
y_t = [np.array(y)/128. for y in y_i]

# for i in range(len(x_t)):
#     print x_t[i], "-", y_t[i]

#     x_t = np.array([[7, 12, 7, 0.717571, -0.0536794, -0.692067, 0.0570439], [7, 12, 7, 0.717489, -0.0537072, -0.692213, 0.0562761], [7, 12, 7, 0.703201, -0.0498395, -0.707105, 0.0550153], [7, 12, 7, 0.703201, -0.0498395, -0.707105, 0.0550153]])
#     y_t = np.array([[161.251, 175.036, 45.1188, 60.7816, 68.9016, 14.7562, 171.948, 57.7966, 91.8903, 59.3473], [161.251, 175.036, 45.1188, 60.7816, 68.9016, 14.7562, 171.948, 57.7966, 91.8903, 59.3473], [173.181, 179.928, 94.0096, 66.7373, 65.5595, 33.9367, 162.365, 83.8356, 42.2113, 59.6372], [173.181, 179.928, 94.0096, 66.7373, 65.5595, 33.9367, 162.365, 83.8356, 42.2113, 59.6372]])/180

# train
with tf.Session() as sess:           
    x = tf.placeholder(tf.float32, shape=[None, max_input_size], name='x')
    y = tf.placeholder(tf.float32, shape=[None, output_size], name='y_in')

    W_h1 = tf.Variable(tf.random_normal([max_input_size, 24]), name='w_h1')
    b_1 = tf.Variable(tf.random_normal([24]), name='b_h1')
    h1 = tf.nn.sigmoid(tf.matmul(x, W_h1) + b_1)

    W_out = tf.Variable(tf.random_normal([24, output_size]), name='w_out')
    b_out = tf.Variable(tf.random_normal([output_size]), name='b_out')
    y_ = tf.nn.sigmoid(tf.matmul(h1, W_out) + b_out, name='y_out')

    cost = tf.reduce_sum( tf.pow(y - y_, 2), 1)
    loss = tf.reduce_mean(cost, name='error')
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, name='train_step')

    sess.run(tf.global_variables_initializer())

    # Saver, to save/load training progress 
    # Can't loads this in C++ (shame), only useful for contining training
    saver = tf.train.Saver() 

    
    for i in range(40):
        sess.run(train_step, feed_dict={x: x_t, y: y_t})

        if i % 20 == 0:
            saver.save(sess, train_file)
            train_error = loss.eval(feed_dict={x: x_t, y: y_t})
            print('step {0}, training error {1}'.format(i, train_error))

    for v in tf.all_variables():
        n = v.name.split(":")[0]    # get name (not sure what the :0 is)
        vc = tf.constant(v.eval(sess), name=n+"_SAVEDCONST")
        tf.assign(v, vc, name=n+"_ASSIGNCONST")

    # Delete output folder if it exists    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    out_fname = out_fname+".pb";
    print("Saving to ", out_path+"/"+out_fname, "...")
    tf.train.write_graph(sess.graph_def, out_path, out_fname, as_text=False)        
    print("...done.")