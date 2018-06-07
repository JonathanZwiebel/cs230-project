import tensorflow as tf
import preprocessor as p

TRAINING_EPOCHS = 100000

LEARNING_RATE = 0.00025

BATCH_SIZE = 1024
SEQUENCE_LENGTH = 100
HIDDEN_LAYER_SIZE = 100

INPUT_SIZE = 192
RNN_OUTPUT_SIZE = HIDDEN_LAYER_SIZE
FINAL_OUTPUT_SIZE = 2


X = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, INPUT_SIZE])
Y = tf.placeholder(dtype=tf.float32, shape=[None, SEQUENCE_LENGTH, FINAL_OUTPUT_SIZE])

def simple_rnn(inputs):
    inputs = tf.unstack(inputs, SEQUENCE_LENGTH, 1)
    rnn_cell = tf.contrib.rnn.LSTMCell(HIDDEN_LAYER_SIZE, activation=tf.nn.tanh)
    intermediate_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell, FINAL_OUTPUT_SIZE)
    rnn_out, hidden_state = tf.contrib.rnn.static_rnn(intermediate_cell, inputs, dtype=tf.float32)
    return rnn_out


predicted_out = simple_rnn(X)
actual_Y = tf.transpose(Y, perm=[1, 0, 2])
loss_op = tf.losses.mean_squared_error(labels=actual_Y, predictions=predicted_out)
tf.summary.histogram('Actual Y', actual_Y)
tf.summary.histogram('Predicted Y', predicted_out)
tf.summary.histogram("Gradients", tf.gradients(loss_op, X))
tf.summary.scalar('loss', loss_op)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)
print("Done making graph")

X_all, Y_all = p.preprocess("/Users/robertross/Documents/CS230-Data/R_2016-01-27_P.mat", "position_relative", seq_length=SEQUENCE_LENGTH)
data_split = p.set_split(X_all, Y_all, {"train": 0.8, "dev": 0.15, "test": 0.05})
dataset = p.Dataset(data_split["train"][0], data_split["train"][1])

merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter("tensorboard/run105", sess.graph)

    batch = 1
    for step in range(TRAINING_EPOCHS):
        loss = -1
        while True:
            print("New batch")
            batch_x, batch_y = dataset.get_next_batch(BATCH_SIZE)
            if batch_x is None:
                print("Epoch: " + str(step) + " | Loss: " + str(loss))
                break
            assert batch_x.shape == (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE)
            assert batch_y.shape == (BATCH_SIZE, SEQUENCE_LENGTH, FINAL_OUTPUT_SIZE)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            summary, loss = sess.run([merged, loss_op], feed_dict={X: batch_x, Y: batch_y})
            train_writer.add_summary(summary, batch)
            batch = batch + 1
            print("Epoch: " + str(step) + " | Loss: " + str(loss))
            break
