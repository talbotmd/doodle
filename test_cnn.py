import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf

def encoder_layer():
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(512,4,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
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
        layer = encoder_layer()
        outputs = layer(outputs)
    for i in range(4):
        layer = decoder_layer(512)
        outputs = layer(outputs)
    layer = decoder_layer(3)
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs,outputs=outputs)

# def Generator():
#     inputs = tf.keras.layers.Input(shape=[256,256,3])  
#     outputs = inputs
#     for i in range(5):
#         layer = encoder_layer()
#         outputs = layer(outputs)
#     return tf.keras.Model(inputs=inputs,outputs=outputs)

def generator_cost(output,target):
    l1_loss = tf.reduce_mean(tf.abs(target - output))
    return l1_loss

def load_dataset():
    train_dataset = h5py.File('data/sketchy/cat.hdf5', "r")
    train_set_x_orig = np.array(train_dataset["image_dataset"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["sketch_dataset"][:]) # your train set labels
    return train_set_x_orig/255, train_set_y_orig/255

def main():
    noise = tf.random.normal([1,256,256,3])
    train_x, train_y = load_dataset()
    generator_model = Generator()
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    for epoch in range(10):
        print("epoch: ", epoch)
        for image in range(train_x.shape[0]):
            print("image: ", image)
            with tf.GradientTape() as gen_tape:
                generated_images = generator_model(train_x[image:image+1],training=True)
                cost = generator_cost(generated_images,train_y[image:image+1])
            gradients_of_generator = gen_tape.gradient(cost, generator_model.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
        output = generator_model(train_x[0:1], training=False)
    
        print("shape:",output.shape)
        plt.imshow(output[0, :, :, :])
        plt.show()

    output = generator_model(train_x[0:1], training=False)
    
    print("shape:",output.shape)
    plt.imshow(output[0, :, :, :])
    plt.show()

    

if __name__ == "__main__":
    main()