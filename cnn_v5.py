import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random

def encoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(num_filters,4,strides=strides,padding='same',kernel_initializer=initializer,use_bias=False))
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(dropout_prob))
    return model

def decoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5, strides=2):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(num_filters,4,strides=strides,padding='same',kernel_initializer=initializer,use_bias=False))
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(dropout_prob))
    return model


def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])  
    outputs = inputs
    layer = encoder_layer(64, apply_dropout=True, dropout_prob=0.1, strides=1)
    outputs = layer(outputs) # 256
    save_1 = outputs
    layer = encoder_layer(128, apply_dropout=True, dropout_prob=0.1, strides=1)
    outputs = layer(outputs)# 256
    save_2 = outputs
    layer = encoder_layer(256, apply_dropout=True, dropout_prob=0.1)
    outputs = layer(outputs) # 128
    save_3 = outputs
    layer = encoder_layer(256, apply_dropout=True, dropout_prob=0.1)
    outputs = layer(outputs) # 64
    save_4 = outputs
    layer = encoder_layer(256, apply_dropout=True, dropout_prob=0.1)
    outputs = layer(outputs) # 32
    save_5 = outputs
        
    # for i in range(2):
    layer = decoder_layer(256, apply_dropout=True, dropout_prob=0.5)
    outputs = layer(outputs) # 65
    outputs = tf.keras.layers.Concatenate()([outputs,save_4])
    layer = decoder_layer(256, apply_dropout=True, dropout_prob=0.5)
    outputs = layer(outputs) # 128
    outputs = tf.keras.layers.Concatenate()([outputs,save_3])
    # for i in range(2):
    layer = decoder_layer(256, apply_dropout=False, strides=2)
    outputs = layer(outputs) # 256
    outputs = tf.keras.layers.Concatenate()([outputs,save_2])
    layer = decoder_layer(256, apply_dropout=False, strides=1)
    outputs = layer(outputs) # 256
    outputs = tf.keras.layers.Concatenate()([outputs,save_1])
    # layer = decoder_layer(3)
    layer = tf.keras.layers.Conv2DTranspose(3,4,strides=1,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
            activation='sigmoid')
    outputs = layer(outputs) # 256
    return tf.keras.Model(inputs=inputs,outputs=outputs)

def Discriminator1():
    inputs = tf.keras.layers.Input(shape=[256,256,3])
    outputs = inputs
    for i in range(2):
        layer = encoder_layer(256, apply_batchnorm=False, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    for i in range(3):
        layer = encoder_layer(128, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    for i in range(2):
        layer = encoder_layer(64, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    layer = tf.keras.layers.Conv2D(1,4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid')
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
    
def Discriminator2():
    inputs = tf.keras.layers.Input(shape=[256,256,6])
    outputs = inputs
    for i in range(2):
        layer = encoder_layer(256, apply_batchnorm=False, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    for i in range(3):
        layer = encoder_layer(128, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    for i in range(2):
        layer = encoder_layer(64, apply_dropout=True, dropout_prob=0.2)
        outputs = layer(outputs)
    layer = tf.keras.layers.Conv2D(1,4,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid')
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

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
def generator_cost(disc2_output,gen_output,target):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    return loss_fn(tf.ones_like(disc2_output),disc2_output) + tf.dtypes.cast(0.01*tf.reduce_mean(tf.abs(target - gen_output)), tf.float32)

def discriminator1_cost(gen_output,human_output):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    gen_loss = loss_fn(tf.zeros_like(gen_output), gen_output)
    human_loss = loss_fn(tf.ones_like(human_output),human_output)
    return gen_loss + human_loss, gen_loss, human_loss

def discriminator2_cost(gen_output,human_output):
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    gen_loss = loss_fn(tf.zeros_like(gen_output), gen_output)
    human_loss = loss_fn(tf.ones_like(human_output),human_output)
    return gen_loss + human_loss, gen_loss, human_loss

def load_dataset(num=4000, start=-1):
    train_dataset = h5py.File('output.hdf5', "r")
    if start < 0:
        start = random.randint(0,9000)
    train_set_x_orig = np.array(train_dataset["image_dataset"][start:start+num],dtype='float32') # your train set features
    train_set_y_orig = np.array(train_dataset["sketch_dataset"][start:start+num],dtype='float32') # your train set labels
    return train_set_x_orig/255, train_set_y_orig/255

def main():
    noise = tf.random.normal([1,256,256,3])
    batch_size = 1
    test_x, test_y = load_dataset(num=100, start=0)
    generator_model = Generator()
    # disc1_model = Discriminator1()
    disc2_model = Discriminator2()
    generator_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.6)
    # disc1_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.3)
    disc2_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.6)

    # Keeps track of the losses for plotting
    gen_losses = []
    # disc1_losses = []
    # disc1_human_losses = []
    # disc1_gen_losses = []
    disc2_losses = []
    disc2_human_losses = []
    disc2_gen_losses = []

    checkpoint_dir = "./checkpoints_v6"
    checkpoint_prefix = os.path.join(checkpoint_dir, "cnn_v6")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, generator=generator_model, disc2_optimizer=disc2_optimizer, disc2=disc2_model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    print("checkpoints: ", manager.checkpoints)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
        for i in range(20):
            # image_x, _ = load_dataset(num=1, start=i)
            print("Restored from {}".format(manager.latest_checkpoint))
            output = generator_model(test_x[i:i+1], training=False)
            print("shape:",output.shape)
            # human_test_gen = np.squeeze(disc1_model(output, training=False))
            match_test_gen = np.squeeze(disc2_model(tf.concat([test_x[i:i+1],output],3), training=False))
            # human_test_human = np.squeeze(disc1_model(test_y[i:i+1], training=False))
            match_test_human = np.squeeze(disc2_model(tf.concat([test_x[i:i+1],test_y[i:i+1]],3), training=False))
            plt.subplot(1,3,1)
            fig = plt.imshow(test_x[i, :, :, :])
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            #plt.show()
            plt.subplot(1,3,2)
            fig = plt.imshow(test_y[i, :, :, :])
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title("Human Doodle\nD_h o/p= "+str(np.round(human_test_human,2))+"\nD_m o/p= "+str(np.round(match_test_human, 2)))
            #plt.show()
            plt.subplot(1,3,3)
            fig = plt.imshow(output[0, :, :, :])
            plt.axis('off')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.title("Generator Output\nD_h o/p= "+str(np.round(human_test_gen,2))+"\nD_m o/p= "+str(np.round(match_test_gen, 2)))
            plt.show()
            #generator_model.save("Generator.h5")
            #disc1_model.save("Discriminator_human.h5")
            #disc2_model.save("Discriminator_match.h5")
        return
        
        # all_losses = np.loadtxt("losses_data")
        # gen_losses = all_losses[0].tolist()
        # disc1_losses = all_losses[1].tolist()
        # disc1_human_losses = all_losses[2].tolist()
        # disc1_gen_losses = all_losses[3].tolist()
        # disc2_losses = all_losses[4].tolist()
        # disc2_human_losses = all_losses[5].tolist()
        # disc2_gen_losses = all_losses[6].tolist()


    

    for epoch in range(3000):
        train_x, train_y = load_dataset(num=500)
        print("epoch: ", epoch)
        average_disc1_cost = 0
        average_human_disc1_cost = 0
        average_gen_disc1_cost = 0
        average_disc2_cost = 0
        average_human_disc2_cost = 0
        average_gen_disc2_cost = 0
        average_gen_cost = 0
        counter = 0
        for image in range(0,train_x.shape[0]-1,batch_size):
            print("image: ", image)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc2_tape:
                # Generate sketches from images
                generated_images = generator_model(train_x[image:min(image+batch_size,train_x.shape[0]-1)],training=True)
                
                train_discs = False
                if counter % 2 == 0:
                    train_discs = True

                #Discriminator 1: Human loss
                # disc1_gen_output = disc1_model(generated_images, training=train_discs)
                # disc1_human_output = disc1_model(train_y[image:min(image+batch_size,train_x.shape[0]-1)], training=train_discs)
                # disc1_cost, disc1_gen_cost, disc1_human_cost = discriminator1_cost(disc1_gen_output, disc1_human_output)

                #Discriminator 2: Matching loss
                disc2_gen_output = disc2_model(tf.concat([train_x[image:min(image+batch_size,train_x.shape[0]-1)],generated_images],3), training=train_discs)
                disc2_human_output = disc2_model(tf.concat([train_x[image:min(image+batch_size,train_x.shape[0]-1)],train_y[image:min(image+batch_size,train_x.shape[0]-1)]],3), training=train_discs)
                disc2_cost, disc2_gen_cost, disc2_human_cost = discriminator2_cost(disc2_gen_output, disc2_human_output)
                
                #Generator loss
                gen_cost = generator_cost(disc2_gen_output,train_x[image:min(image+batch_size,train_x.shape[0]-1)],train_y[image:min(image+batch_size,train_x.shape[0]-1)])
                
                #Tracking loss
                # average_human_disc1_cost += disc1_human_cost
                # average_gen_disc1_cost += disc1_gen_cost
                # average_disc1_cost += disc1_cost
                average_human_disc2_cost += disc2_human_cost
                average_gen_disc2_cost += disc2_gen_cost
                average_disc2_cost += disc2_cost
                average_gen_cost += gen_cost
                counter += 1

            ### Gradient Decent ###

            #Generator
            gradients_of_generator = gen_tape.gradient(gen_cost, generator_model.trainable_variables)
            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
            
            # #Discriminator 1: Human
            # gradients_of_disc1 = disc1_tape.gradient(disc1_cost, disc1_model.trainable_variables)
            # disc1_optimizer.apply_gradients(zip(gradients_of_disc1,disc1_model.trainable_variables))
        
            #Discriminator 2: Matching
            gradients_of_disc2 = disc2_tape.gradient(disc2_cost, disc2_model.trainable_variables)
            disc2_optimizer.apply_gradients(zip(gradients_of_disc2,disc2_model.trainable_variables))


        # print("disc1 cost: ", average_disc1_cost/counter)
        # print("human disc1 cost: ", average_human_disc1_cost/counter)
        # print("gen disc1 cost: ", average_gen_disc1_cost/counter)
        print("disc2 cost: ", average_disc2_cost/counter)
        print("human disc2 cost: ", average_human_disc2_cost/counter)
        print("gen disc2 cost: ", average_gen_disc2_cost/counter)
        print("gen cost: ", average_gen_cost/counter)
        gen_losses.append(np.mean(average_gen_cost)/counter)
        # disc1_losses.append(np.mean(average_disc1_cost)/counter)
        # disc1_human_losses.append(np.mean(average_human_disc1_cost)/counter)
        # disc1_gen_losses.append(np.mean(average_gen_disc1_cost)/counter)
        disc2_losses.append(np.mean(average_disc2_cost)/counter)
        disc2_human_losses.append(np.mean(average_human_disc2_cost)/counter)
        disc2_gen_losses.append(np.mean(average_gen_disc2_cost)/counter)
        

        np.savetxt('losses_data',np.array([gen_losses,disc2_losses,disc2_human_losses,disc2_gen_losses]))

        # Plot loss and save train/test images on every iteration
        output = generator_model(test_x[test_x.shape[0]-1:test_x.shape[0]], training=False)
        plt.imshow(output[0, :, :, :])
        plt.savefig("test-" + str(len(gen_losses)) + ".png")
        plt.clf()
        output = generator_model(test_x[0:1], training=False)
        plt.imshow(output[0, :, :, :])
        plt.savefig("train-" + str(len(gen_losses)) + ".png")
        plt.clf()
        plt.plot(gen_losses)
        plt.savefig("gen_losses.png")
        plt.clf()
        # plt.plot(disc1_losses)
        # plt.savefig("disc1_losses.png")
        # plt.clf()
        # plt.plot(disc1_gen_losses)
        # plt.savefig("disc1_gen_losses.png")
        # plt.clf()
        # plt.plot(disc1_human_losses)
        # plt.savefig("disc1_human_losses.png")
        # plt.clf()
        plt.plot(disc2_losses)
        plt.savefig("disc2_losses.png")
        plt.clf()
        plt.plot(disc2_gen_losses)
        plt.savefig("disc2_gen_losses.png")
        plt.clf()
        plt.plot(disc2_human_losses)
        plt.savefig("disc2_human_losses.png")
        plt.clf()

	    # plt.imshow(output[0, :, :, :])
            # plt.show()

        # Save a checkpoint every other iteration
        if epoch % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)



    output = generator_model(test_x[0:1], training=False)
    
    print("shape:",output.shape)
    plt.imshow(output[0, :, :, :])
    plt.show()

    

if __name__ == "__main__":
    main()
