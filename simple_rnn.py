import tensorflow as tf
import preprocessor as p

TRAINING_EPOCHS = 1000
PRITNING_EPOCHS = 10

LEARNING_RATE = 0.001

BATCH_SIZE = 256
SEQUENCE_LENGTH = 350
HIDDEN_LAYER_SIZE = 300

INPUT_SIZE = 192
RNN_OUTPUT_SIZE = HIDDEN_LAYER_SIZE
INTERMEDIATE_OUTPUT_SIZE = 25
FINAL_OUTPUT_SIZE = 2


X = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, INPUT_SIZE])
Y = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, FINAL_OUTPUT_SIZE])

def simple_rnn(inputs):
    inputs = tf.unstack(inputs, SEQUENCE_LENGTH, 1)
    rnn_cell = tf.contrib.rnn.BasicRNNCell(HIDDEN_LAYER_SIZE, activation=tf.nn.tanh)
    intermediate_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell, FINAL_OUTPUT_SIZE)
    rnn_out, hidden_state = tf.contrib.rnn.static_rnn(intermediate_cell, inputs, dtype=tf.float32)
    return rnn_out


predicted_out = simple_rnn(X)
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=tf.transpose(Y, perm=(1, 0, 2)), predictions=predicted_out))
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)
print("Done making graph")

X_all, Y_all = p.preprocess("D:\cs230\R_2016-01-27_P", "position_relative", seq_length=800)
data_split = p.set_split(X_all, Y_all, {"train": 0.8, "dev": 0.15, "test": 0.05})
dataset = p.Dataset(data_split["train"][0], data_split["train"][1])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(TRAINING_EPOCHS):
        loss = -1
        while True:
            print("New batch")
            batch_x, batch_y = dataset.get_next_batch(BATCH_SIZE)
            if batch_x is None:
                print("Epoch: " + str(step) + " | Loss: " + str(loss))
                break
            assert batch_x.shape == (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y})
