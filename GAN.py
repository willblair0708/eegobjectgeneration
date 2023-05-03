import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def one_hot(y_):
    y_ = y_.reshape(len(y_))
    y_ = [int(x) for x in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def get_generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    global n_class
    momentum = 0.9

    with tf.variable_scope("generator", reuse=reuse):
        # Deconvolutional
        noise_img = tf.keras.layers.Dense(units=140 * 64)(noise_img)

        h0 = tf.reshape(noise_img, [-1, 10, 14, 64])

        h0 = tf.nn.relu(tf.keras.layers.BatchNormalization(momentum=momentum)(h0))

        h3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=5, strides=2, padding='same')(h0)
        h3 = tf.nn.relu(tf.keras.layers.BatchNormalization(momentum=momentum)(h3))

        h4 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=5, strides=2, padding='same', name='g')(h3)

        # logits & outputs
        logits = tf.reshape(h4, [-1, 40 * 56])
        outputs = tf.math.tanh(logits)
        mid = tf.math.sigmoid(tf.keras.layers.Dense(units=20 * 20)(logits))
        pred = tf.keras.layers.Dense(units=n_class)(mid)

        return logits, outputs, pred

def get_discriminator(img, n_units, reuse=False, alpha=0.01, cond=None):
    momentum = 0.9
    con = cond[:, noise_size - 40:noise_size]

    with tf.variable_scope("discriminator", reuse=reuse):
        # CNN
        z_image = tf.reshape(img, [-1, 40, 56, 1])
        h0 = lrelu(tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=2, padding='same')(z_image))
        h0 = lrelu(tf.keras.layers.BatchNormalization(momentum=momentum)(h0))

        h1 = lrelu(tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=2, padding='same')(h0))
        h1 = lrelu(tf.keras.layers.BatchNormalization(momentum=momentum)(h1))

        hidden1 = tf.keras.layers.Flatten()(h1)

        hidden1 = tf.concat([hidden1, con], axis=1)

        logits = tf.keras.layers.Dense(units=1)(hidden1)
        outputs = logits
        outputs_2 = tf.keras.layers.Dense(units=n_class)(hidden1)

        return logits, outputs, outputs_2
    
def compute_accuracy(v_xs, v_ys):
    correct_prediction = tf.equal(tf.argmax(v_xs, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

img_size = 2205
noise_size = 60
n_class = 5

"""
load the shuffled data, [3200, 2281], 2286 = 40 + 2240 + 1
In which, 40 is EEG features, 2240 = 40*56 is the image, 1 is the label
"""
data = pickle.load(open('shape_EEG_feature.pkl', 'rb'))
print(data.shape)

label = data[:, -1]
label.shape = [3200, 1]
# print(np.sum(label), label.shape, np.max(data[0, 40:-1]), np.min(data[0, 40:-1]))

label = tf.one_hot(label, n_class)

# make the D_2 label
g_units = 200
d_units = g_units
alpha = 0.01
# label smoothing
smooth = 0.1

tf.keras.backend.clear_session()
real_img = tf.compat.v1.placeholder(tf.float32, [None, img_size], name='real_img')
noise_img = tf.compat.v1.placeholder(tf.float32, [None, noise_size], name='noise_img')
ground_truth = tf.compat.v1.placeholder(tf.float32, shape=[None, n_class], name='ground_truth')
print(real_img.shape)
print(noise_img.shape)
print(ground_truth.shape)

real_img_reshaped = tf.reshape(real_img, [-1, 2240])
print(real_img_reshaped.shape)

# generator
g_logits, g_outputs, pred = get_generator(noise_img, g_units, img_size)
# discriminator
d_logits_real, d_outputs_real, real_category_pred = get_discriminator(real_img, d_units, cond=noise_img)
d_logits_fake, d_outputs_fake, fake_category_pred = get_discriminator(g_outputs, d_units, cond=noise_img, reuse=True)

# ACC
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(ground_truth, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# # discriminatorloss # cross-entropy
d_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_logits_real)) * (1 - smooth))
d_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)))

# reduce_mean
d_loss_rf = d_loss_real + d_loss_fake

d_loss_category_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=real_category_pred, labels=ground_truth))
d_loss_category_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_category_pred, labels=ground_truth))
d_loss_category = 0.8 * d_loss_category_real + d_loss_category_fake
d_loss = d_loss_rf + d_loss_category_real

# classifier loss
c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=ground_truth))
# calculate the inception classification accuracy, evaluating is the generated image is correct?
IC_fake = compute_accuracy(ground_truth, fake_category_pred)
IC_real = compute_accuracy(ground_truth, real_category_pred)

# reshape the array
#arr.reshape((-1, 2400))

# generator loss
lambda_ = 0.01
batch_size = 80

g_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)) * (1 - smooth))

# regularization of generated fake image with the real image:
g_regular = tf.keras.losses.mean_squared_error(y_true=real_img_reshaped, y_pred=g_outputs)
g_loss = g_loss + d_loss_category_fake + lambda_ * g_regular

train_vars = tf.trainable_variables()

# generator tensor
g_vars = [var for var in train_vars if var.name.startswith("generator")]
# discriminator tensor
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

# optimizer
learning_rate = 0.0002  # 0.0002
d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
c_optimizer = tf.train.AdamOptimizer(learning_rate)

# batch_size
epochs = 200
n_sample = batch_size
n_batch = int(data.shape[0] / batch_size)

samples = []
losses = []

saver = tf.train.Saver(var_list=g_vars)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

tf.compat.v1.disable_eager_execution()

# Define discriminator optimizer
d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
d_gradients = tf.gradients(d_loss, d_vars)
d_train_opt = d_optimizer.apply_gradients(zip(d_gradients, d_vars))

# Define generator optimizer
g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
g_gradients = tf.gradients(g_loss, g_vars)
g_train_opt = g_optimizer.apply_gradients(zip(g_gradients, g_vars))

# Initialize global variables
tf.compat.v1.global_variables_initializer().run(session=sess)

for e in range(epochs):
    for h in range(n_batch):
        # print("this is number ", h)
        z_batch = data[batch_size * h:batch_size * (h + 1), 0:40]
        real_image_batch = data[batch_size * h:batch_size * (h + 1), 40:-1]
        
        label_batch = label[batch_size * h:batch_size * (h + 1)]
        label_batch = tf.cast(label_batch, tf.float32)
        label_batch_tensor = tf.convert_to_tensor(label_batch)
        label_batch_tensor = tf.cast(label_batch_tensor, dtype=tf.float32)

        # batch_images = batch[0].reshape((batch_size, 784))
        real_image_batch = real_image_batch * 2 - 1
        z_batch = z_batch * 2 - 1

        # reshape the real_image_batch to (?, 2240)
        #real_image_batch = tf.reshape(real_image_batch, (-1, 2240))

        batch_noise = tf.random.uniform([batch_size, noise_size], minval=-1, maxval=1)
        batch_noise = tf.concat([batch_noise[:, :20], z_batch[:, :40]], axis=1)  # 130 noise + 10 EEG channels
        # batch_noise = z_batch[:, :40]
        batch_noise_tensor = tf.cast(batch_noise, dtype=tf.float32)

        # Convert real_image_batch to a NumPy ndarray
        with tf.compat.v1.Session() as sess1:
            real_image_batch_np = sess1.run(tf.identity(real_image_batch))

        #print(real_img.shape, real_image_batch.shape)
        # Run optimizers
        #real_image_batch_tensor = tf.convert_to_tensor(real_image_batch)
        #label_batch_tensor = tf.convert_to_tensor(label_batch)
        #batch_noise_tensor = tf.convert_to_tensor(batch_noise, dtype=tf.float32)
        sess.run(d_train_opt, feed_dict={real_img: real_image_batch_np, noise_img: batch_noise_tensor, ground_truth: label_batch_tensor})
        for i in range(2):
            _ = tf.compat.v1.keras.backend.get_session().run(g_train_opt,
                                                            feed_dict={real_img: real_image_batch, noise_img: batch_noise, ground_truth: label_batch})

    if e % 50 == 0 and e != 0:
        # discriminator loss
        train_loss_d, train_loss_d_rf, train_loss_d_category = tf.compat.v1.Session().run([d_loss, d_loss_rf, d_loss_category],
                                                                        feed_dict={real_img: real_image_batch,
                                                                                   noise_img: batch_noise,
                                                                                   ground_truth: label_batch})

        # generator loss
        train_loss_g, train_loss_c, acc, g_regular_ = tf.compat.v1.Session().run([g_loss, c_loss, accuracy, g_regular],
                                                               feed_dict={real_img: real_image_batch,
                                                                          noise_img: batch_noise,
                                                                          ground_truth: label_batch})
        # IC score
        ic_real_, ic_fake_ = tf.compat.v1.Session().run([IC_real, IC_fake],
                                      feed_dict={real_img: real_image_batch, noise_img: batch_noise,
                                                 ground_truth: label_batch})
        print("Epoch {}/{}".format(e + 1, epochs),
              "D Loss: {:.4f}(r/f: {:.4f} + category: {:.4f})".format(train_loss_d, train_loss_d_rf,
                                                                      train_loss_d_category),
              "G Loss: {:.4f}, RMSE:{:.4f} , C loss{:.4f}, acc, {:.4f}".format(train_loss_g,
                                                                    lambda_ * g_regular_, train_loss_c, acc),
              'IC real.{:.4f}, fake: {:.4f}'.format(ic_real_, ic_fake_))
        losses.append((train_loss_d, train_loss_d_rf, train_loss_d_category, train_loss_g), )

        # fig = plt.figure(figsize=(30, 6))
        # print 'true label,', true_label[1:10]
        # no_pic = 12
        # for i in range(1, no_pic+1):  # 8 samples including 5 categories
        #     generated_image = gen_samples[i].reshape([40, 56])
        #     real_image = real_[i].reshape([40, 56])
        #     fig.add_subplot(1, no_pic, i)
        #     plt.axis('off')
        #     plt.imshow(generated_image, cmap='gray_r')
        # plt.savefig('generated_images/step'+str(e)+'.png', format='png', bbox_inches='tight')
        # pickle.dump(gen_samples, open('GAN_1.pk', 'wb'))

        #Test
        h = n_batch - 3
        z_batch_ = data[batch_size * h:batch_size * (h + 1), :40]  # the last batch worked as testingsample
        true_label = data[batch_size * h:batch_size * (h + 1), -1]
        real_ = data[batch_size * h:batch_size * (h + 1), 40:-1]
        #half noise half EEG
        sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))
        sample_noise = np.hstack((sample_noise[:, :20], z_batch_[:, :40]))  # 130 noise + 10 EEG channels

        with tf.compat.v1.variable_scope('generator', reuse=True):
            w1 = tf.compat.v1.get_variable('w1')
            b1 = tf.compat.v1.get_variable('b1')
            w2 = tf.compat.v1.get_variable('w2')
            b2 = tf.compat.v1.get_variable('b2')
            hidden = tf.nn.relu(tf.matmul(noise_img, w1) + b1)
            output = tf.nn.tanh(tf.matmul(hidden, w2) + b2)

        _, gen_samples, pred_ = tf.compat.v1.Session().run((w1, w2, b1, b2, output),
                                         feed_dict={real_img: real_, noise_img: sample_noise, })

