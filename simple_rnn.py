import tensorflow as tf
import preprocessor as p
import numpy as np
import sys

RUN_VALUE = int(sys.argv[1])

TRAINING_EPOCHS = 100000
PLOTTING_EPOCHS = 10
TEST_EPOCHS = 5

LEARNING_RATE = 0.001

BATCH_SIZE = 1024
SEQUENCE_LENGTH = 500
SKIP_LENGTH = 100
HIDDEN_LAYER_SIZE = 50

INPUT_SIZE = 192
RNN_OUTPUT_SIZE = HIDDEN_LAYER_SIZE
INTERMEDIATE_SIZE = 30
FINAL_OUTPUT_SIZE = 2


datafiles = ['R_2016-01-27_P']  # , 'R_2016-01-28_P', 'R_2016-01-29_P', 'R_2016-02-02_P', 'R_2017-11-02_P']
X = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, INPUT_SIZE])
Y = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, FINAL_OUTPUT_SIZE])

def simple_rnn(inputs):
    inputs = tf.unstack(inputs, SEQUENCE_LENGTH, 1)
    rnn_cell = tf.contrib.rnn.LSTMCell(HIDDEN_LAYER_SIZE, activation=tf.nn.tanh)
    intermediate_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell, INTERMEDIATE_SIZE, activation=tf.nn.relu)
    rnn_out, hidden_state = tf.contrib.rnn.static_rnn(intermediate_cell, inputs, dtype=tf.float32)
    final_out = tf.contrib.layers.fully_connected(rnn_out, FINAL_OUTPUT_SIZE, activation_fn=None)
    return final_out


predicted_out = simple_rnn(X)
tf.shape(predicted_out)
actual_Y = tf.transpose(Y, perm=[1, 0, 2])
loss_op = tf.losses.mean_squared_error(labels=actual_Y, predictions=predicted_out)
tf.summary.scalar('loss', loss_op)

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
gvs = optimizer.compute_gradients(loss_op)
capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs)
print("Done making graph")

X_all, Y_all = p.preprocess(datafiles, "position_relative", seq_length=SEQUENCE_LENGTH, skip_length=SKIP_LENGTH)
data_split = p.set_split(X_all, Y_all, {"train": 0.8, "dev": 0.15, "test": 0.05})
print(len(data_split["train"][0]))
print(len(data_split["dev"][0]))
print(len(data_split["test"][0]))
dataset = p.Dataset(data_split["train"][0], data_split["train"][1])

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("tensorboard/run" + str(RUN_VALUE), sess.graph)
    test_writer = tf.summary.FileWriter("tensorboard/test_run" + str(RUN_VALUE), sess.graph)

    batch = 1
    for step in range(TRAINING_EPOCHS):
        loss = -1
        while True:
            print("New batch")
            batch_x, batch_y = dataset.get_next_batch(BATCH_SIZE)
            # if batch_x is None:
            #     print("Epoch: " + str(step) + " | Loss: " + str(loss))
            #     break
            assert batch_x.shape == (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
            assert batch_y.shape == (BATCH_SIZE, SEQUENCE_LENGTH, FINAL_OUTPUT_SIZE)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            summary, loss = sess.run([merged, loss_op], feed_dict={X: batch_x, Y: batch_y})

            if step % PLOTTING_EPOCHS == 0:
                print("Plotting values")
                predictions, true = sess.run([predicted_out, actual_Y], feed_dict={X: batch_x, Y: batch_y})
                np.save("output/run" + str(RUN_VALUE) + "/pred_epoch" + str(step), predictions)
                np.save("output/run" + str(RUN_VALUE) + "/true_epoch" + str(step), true)

            if step % TEST_EPOCHS == 0:
                print("Calculating test loss")
                test_loss = sess.run([loss_op], feed_dict={X: data_split["dev"][0], Y: data_split["dev"][1]})
                test_summary = tf.Summary()
                test_summary.value.add(tag='test_loss', simple_value=test_loss[0])
                test_writer.add_summary(test_summary, batch)

            train_writer.add_summary(summary, batch)
            batch = batch + 1
            print("Epoch: " + str(step) + " | Loss: " + str(loss))
            break
