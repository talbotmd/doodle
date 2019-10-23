import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os

def encoder_layer(num_filters):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(num_filters,4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model

def decoder_layer(num_filters):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(num_filters,4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    return model


def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])  
    outputs = inputs
    for i in range(5):
        layer = encoder_layer(256)
        outputs = layer(outputs)
    for i in range(4):
        layer = decoder_layer(256)
        outputs = layer(outputs)
    layer = decoder_layer(3)
    # layer = tf.keras.layers.Conv2DTranspose(3,4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
    #         activation='tanh')
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs,outputs=outputs)

def Discriminator1():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    ouputs = inputs
    for i in range(5):
        layer = encoder_layer(256)
        outputs = layer(outputs)
    layer = tf.keras.layers.Conv2d(1,4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid')
    
def Discriminator2():
    inputs = tf.keras.layers.Input(shape=[256,256,6])
    ouputs = inputs
    for i in range(5):
        layer = encoder_layer(256)
        outputs = layer(outputs)
    layer = tf.keras.layers.Conv2d(1,4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid')

# def Generator():
#     inputs = tf.keras.layers.Input(shape=[256,256,3])  
#     outputs = inputs
#     for i in range(5):
#         layer = encoder_layer()
#         outputs = layer(outputs)
#     return tf.keras.Model(inputs=inputs,outputs=outputs)

# def generator_cost(output,target):
    # l1_loss = tf.reduce_mean(tf.abs(target - output))
    # return l1_loss
def generator_cost(disc1_output):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    return loss_fn(tf.ones_like(disc1_output), disc1_output)# + loss_fn(tf.ones_like(disc2_output),disc2_output)

def discriminator1_cost(gen_output,human_output):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    return loss_fn(tf.zeros_like(gen_output), gen_output) + loss_fn(tf.ones_like(human_output),human_output)

def discriminator2_cost(gen_output,human_output):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    return loss_fn(tf.zeros_like(gen_output), gen_output) + loss_fn(tf.ones_like(human_output),human_output)

def load_dataset():
    train_dataset = h5py.File('output.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["image_dataset"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["sketch_dataset"][:]) # your train set labels
    return train_set_x_orig/255, train_set_y_orig/255

def main():
    noise = tf.random.normal([1,256,256,3])
    train_x, train_y = load_dataset()
    generator_model = Generator()
    disc1_model = Discriminator1()
    disc2_model = Discriminator2()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc1_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = "./checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "cat_chpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator_model, discriminator_optimizer=disc1_optimizer,discriminator=disc1_model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        output = generator_model(train_x[94:95], training=False)
        print("shape:",output.shape)
        plt.imshow(output[0, :, :, :])
        plt.show()
        return

    for epoch in range(3000):
        print("epoch: ", epoch)
        for image in range(0,train_x.shape[0]-1,5):
            print("image: ", image)
            with tf.GradientTape() as gen_tape:
                generated_images = generator_model(train_x[image:min(image+5,train_x.shape[0]-1)],training=True)
                disc1_gen_output = disc1_model(generated_images, training=True)
                disc1_human_output = disc1_model(train_y[image:min(image+5,train_x.shape[0]-1)], training=True)
                # disc2_gen_output = disc2_model(generated_images)
                # disc2_human_output = disc2_model(train_y[image:min(image+5,train_x.shape[0]-1)])
                disc1_cost = discriminator1_cost(disc1_gen_output, disc1_human_output)
                # disc2_cost = discriminator2_cost(disc2_gen_output, disc2_human_output)
                gen_cost = generator_cost(disc1_cost)
                
            gradients_of_generator = gen_tape.gradient(gen_cost, generator_model.trainable_variables)
            gradients_of_disc1 = gen_tape.gradient(disc1_cost, disc1_model.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
            disc1_optimizer.apply_gradients(zip(gradients_of_disc1,disc1_model.trainable_variables))
        # with tf.GradientTape() as gen_tape:
        #     generated_images = generator_model(train_x,training=True)
        #     cost = generator_cost(generated_images,train_y)
        # gradients_of_generator = gen_tape.gradient(cost, generator_model.trainable_variables)
        # generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
        
        if epoch % 100 == 0:
            output = generator_model(train_x[train_x.shape[0]-1:train_x.shape[0]], training=False)
            print("shape:",output.shape)
            plt.imshow(output[0, :, :, :])
            plt.show()
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)



    output = generator_model(train_x[0:1], training=False)
    
    print("shape:",output.shape)
    plt.imshow(output[0, :, :, :])
    plt.show()

    

if __name__ == "__main__":
    main()