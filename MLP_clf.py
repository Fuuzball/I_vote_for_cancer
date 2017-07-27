import numpy as np
import tensorflow as tf

np.set_printoptions(precision=2)
X_train = np.load('./data/train_doc.npy')
y_train = np.load('./data/train_label.npy')
X_test = np.load('./data/test_doc.npy')
y_test = np.load('./data/test_label.npy')

n_input = 100
n_classes = 9
train_size = X_train.shape[0]

def get_batch(n_batch):
    train_indices = np.arange(train_size)
    np.random.shuffle(train_indices)
    idx = train_indices[:n_batch]
    return X_train[idx], y_train[idx]

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Training params
learning_rate = 0.005
training_epochs = 500
batch_size = 100
display_epoch = 10

# Network Params
n_hidden = 25

def test_model(learning_rate, training_epochs, batch_size, n_hidden):
    # Construct model
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden]))
    b1 = tf.Variable(tf.random_uniform([n_hidden]))

    W2 = tf.Variable(tf.random_uniform([n_hidden, n_classes]))
    b2 = tf.Variable(tf.random_uniform([n_classes]))

    W = tf.Variable(tf.random_uniform([n_input, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    u = tf.nn.relu(tf.matmul(x, W1) + b1)
    pred = tf.nn.softmax(tf.matmul(u, W2) + b2)
    pred = tf.nn.softmax(tf.matmul(x, W) + b)
    x_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 10**(-15)), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(x_entropy)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(train_size // batch_size)
            train_indices = np.arange(train_size)
            np.random.shuffle(train_indices)
            for i in range(total_batch):
                idx = train_indices[i * batch_size:(i + 1) * batch_size]
                batch_x, batch_y =  X_train[idx], y_train[idx]
                _, c = sess.run([train_step, x_entropy], feed_dict = {x:batch_x, y:batch_y})

                avg_cost += c / total_batch

            if (epoch+1) % display_epoch == 0: 
                print('Epoch: ', epoch + 1, 'cost: ', avg_cost)

                log_loss = sess.run(x_entropy, feed_dict={x:X_test, y:y_test})
                print('Log Loss: ', log_loss)

        log_loss = sess.run(x_entropy, feed_dict={x:X_test, y:y_test})
        corr_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(corr_pred, 'float'))
        print('---------')
        print('Log Loss: ', log_loss)
        print('Acuracy: ', sess.run(acc, feed_dict={x:X_test, y:y_test}))
        pred_sample = sess.run(pred, feed_dict={x:X_test[:5]})
        print(np.round(pred_sample, 2))
        print(y_test[:5])
    return log_loss

test_model(learning_rate, training_epochs, batch_size, n_hidden)
