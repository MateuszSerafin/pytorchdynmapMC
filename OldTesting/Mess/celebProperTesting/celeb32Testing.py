import bz2
import pickle

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import time
from PIL import Image
import keras.layers as layers
import tqdm


BATCH_SIZE = 128

EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 25


def mnist_desc():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.burdel = "mnist_desc"
    return model

def copied_desc():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(3, (4, 4), strides=(1, 1), padding='same', input_shape=[64, 64, 3]))

    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(1024, (4, 4), strides=(1, 1), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.burdel = "copied_desc"
    return model

def copied_desc_withoutfirstlayer():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(1024, (4, 4), strides=(1, 1), padding='valid'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.burdel = "copied_desc_without_firstlayer"
    return model

def copied_desc_gigasize():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(3, (4, 4), strides=(2, 2), padding='same', input_shape=[64, 64, 3]))
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1024, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(2048, (4, 4), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.burdel = "copied_desc_gigasize"
    return model



def mnist_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    model.burdel = "mnist_generator"
    return model

def mnist_generator_tiple():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Reshape((8, 8, 1024)))
    assert model.output_shape == (None, 8, 8, 1024)  # Note: None is the batch size
    model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    model.burdel = "mnist_generator_triple"
    return model

def propergenerator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(2 * 2 * 1024, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((2, 2, 1024)))
    assert model.output_shape == (None, 2, 2, 1024)

    model.add(layers.Conv2DTranspose(1024, (4, 4), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(512, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 4, 4, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())


    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 64, 64, 3)

    model.burdel = "propergenerator"

    return model


generators = [mnist_generator, mnist_generator_tiple, propergenerator]
descriminators = [mnist_desc, copied_desc, copied_desc_gigasize, copied_desc_withoutfirstlayer]



noise = tf.random.normal([1, 100])
for generator in generators:
    generated = generator()(noise)
    for discriminator in descriminators:
        discriminator()(generated)
        #doesnt error = gud

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


seed = tf.random.normal([num_examples_to_generate, noise_dim])





def generate_and_save_images(descriminator, model, epoch, test_input):
    if(not os.path.exists(descriminator.burdel)):
        os.mkdir(descriminator.burdel)
    maindir = os.path.join(descriminator.burdel, model.burdel)

    if(not os.path.exists(maindir)):
        os.mkdir(maindir)

    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(5, 5))

    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        plt.imshow(((predictions[i].numpy() + 1) * 127.5).astype(np.uint8))
        plt.axis('off')

    plt.savefig(maindir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()


def train(epochs):
    for descriminator in range(len(descriminators)):
        for generatorint in range(len(generators)):
            generator_glob = generators[generatorint]()
            discriminator_glob = descriminators[descriminator]()

            discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
            generator_optimizer = tf.keras.optimizers.Adam(1e-4)
            try:
                for epoch in range(epochs):
                    start = time.time()

                    file = open("nowater.pickled", "rb")

                    try:
                        while True:
                            loaded = pickle.load(file)

                            BUFFER_SIZE = len(loaded)
                            train_dataset = tf.data.Dataset.from_tensor_slices(loaded).shuffle(
                                BUFFER_SIZE).batch(BATCH_SIZE)

                            @tf.function
                            def train_step(images):
                                noise = tf.random.normal([BATCH_SIZE, noise_dim])

                                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                                    generated_images = generator_glob(noise, training=True)

                                    real_output = discriminator_glob(images, training=True)
                                    fake_output = discriminator_glob(generated_images, training=True)

                                    gen_loss = generator_loss(fake_output)
                                    disc_loss = discriminator_loss(real_output, fake_output)

                                gradients_of_generator = gen_tape.gradient(gen_loss, generator_glob.trainable_variables)
                                gradients_of_discriminator = disc_tape.gradient(disc_loss,
                                                                                discriminator_glob.trainable_variables)

                                generator_optimizer.apply_gradients(
                                    zip(gradients_of_generator, generator_glob.trainable_variables))
                                discriminator_optimizer.apply_gradients(
                                    zip(gradients_of_discriminator, discriminator_glob.trainable_variables))

                            for image_batch in tqdm.tqdm(train_dataset):
                                train_step(image_batch)

                    except (EOFError):
                        pass

                    file.close()


                    # Produce images for the GIF as you go
                    generate_and_save_images(discriminator_glob, generator_glob,
                                             epoch + 1,
                                             seed)

                    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

                generate_and_save_images(discriminator_glob, generator_glob,
                                         epochs,
                                         seed)
            except KeyboardInterrupt as e:
                continue
train(300)