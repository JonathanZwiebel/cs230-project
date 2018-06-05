import tensorflow as tf
import preprocessor

HIDDEN_SIZE = 250
BATCH_SIZE = 256
INPUT_SIZE = 192
OUTPUT_SIZE = 2
SEQUENCE_LENGTH = 800
RNN_ACTIVATION = tf.nn.tanh


# TODO: Implement regularization
# TODO: Implement normalization
# TODO: Implement LR decay


class BasicRNNModel:
    def __init__(self):
        self.optimizer = tf.train.AdamOptimizer()
        self.X_train = tf.placeholder(tf.int32, [BATCH_SIZE, SEQUENCE_LENGTH])
        self.Y_train = tf.placeholder(tf.float32, [BATCH_SIZE, SEQUENCE_LENGTH])
        self.initial_state = None
        self.final_state = None
        self.cost = None
        self.predictions = None
        self.train_op = None

    def create_network(self):
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE, activation=RNN_ACTIVATION)
        self.initial_state = rnn_cell.zero_state(BATCH_SIZE, tf.float32)

        outputs = []
        state = self.initial_state

        # TODO: Make inputs

        for i in range(SEQUENCE_LENGTH):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = rnn_cell(inputs[:, i, :], state)
            outputs.append[output]
        self.final_state = state

        output_w = tf.get_variable("output w", [BATCH_SIZE, 3])
        output_b = tf.get_variable("output b", [3])
        self.predictions = tf.matmul(outputs, output_w) + output_b

        loss = -1# TODO: Loss
        self.cost = tf.reduce_sum(loss) / BATCH_SIZE

        training_variables = tf.trainable_variables()
        gradient, _ = tf.gradients(cost, trainable_variables)
        self.train_op = self.optimizer.apply_gradients(zip(gradient, training_variables))

X_train, Y_train = preprocessor.preprocess("D:\cs230\JR_2015-12-04_truncated2.mat")
create_network(X_train, Y_train)
run_network(X_train, Y_train)
