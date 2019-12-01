import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import random

def encoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(num_filters,3,strides=1,padding='same',kernel_initializer=initializer,use_bias=False))
    model.add(tf.keras.layers
        .MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
        
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(dropout_prob))
    return model
    
def decoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2DTranspose(num_filters,3,strides=2,padding='same',kernel_initializer=initializer,use_bias=False))
    
    if apply_batchnorm:
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU())
    if apply_dropout:
        model.add(tf.keras.layers.Dropout(dropout_prob))
    return model

def EncoderP2E():
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
    layer = tf.keras.layers.Conv2D(1,3,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid') #Make fully connected
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name = "EncoderP2E")
    
def EncoderD2E():
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
    layer = tf.keras.layers.Conv2D(1,3,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid') #Make fully connected
    outputs = layer(outputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs,  name = "EncoderD2E")
    
    

def load_minibatch(num=1000, start=-1, n=5):
    #Assuming this loads a minibatch of 1000 (picture + doodles + augment pairs) * 3 (examples positive and negative, indicated by y_train)
    # Assumes the in output.hdf5 pairs are already shuffled and randomized. negative examples are created by adding a
    train_dataset = h5py.File('output.hdf5', "r")
    if start < 0:
        start = random.randint(0,2000-n)
    train_set_P_orig = np.array(train_dataset["image_dataset"][start:start+num],dtype='float32')
    train_set_D_orig = np.array(train_dataset["sketch_dataset"][start:start+num],dtype='float32')
    
    train_set_P = train_set_P_orig
    train_set_D = train_set_D_orig
    y_train = np.ones([num,1],dtype='float32')
    
    #for i in range(1, n):
    train_set_D_orig_false =  np.array(train_dataset["bad_sketch_dataset"][start:start+num],dtype='float32') # your train set labels
    train_set_P = tf.concat([train_set_P, train_set_P_orig], 0)
    train_set_D = tf.concat([train_set_D, train_set_D_orig_false], 0)
    y_train = tf.concat([y_train, np.zeros([num,1],dtype='float32')],axis=0)
    print("y_train:",y_train.shape)
    return train_set_P/255, train_set_D/255, y_train


def EmbeddingCost(E_P, E_D, y_train,margin, n):
    # E_P is embedding output of picture, E_D is embedding output of Doodle, y_train indiciates whether it is positive or negative example, margin is triplet margin, n is number fo negaitve examples per positive example
    
    # Theres a problem with this.  I think we need to use a keras layer to compute the cost
    m = len(y_train)
    #loss_fn = 1/m*(tf.keras.backend.sum(((1+1/n)*y_train - 1/n)*tf.norm(E_D-E_P,axis=1,ord=2)))+margin
    distance_norm = tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(E_D - E_P), axis = 2))
    loss_func = tf.keras.backend.sum(tf.keras.backend.relu(tf.multiply(ditance_norm,1/m*((1+1/n)*y_train - 1/n))+margin))
        # This should give norm(ED-EP) when y = 1, and 1/n*norm(ED-EP) when y is 0.
    return loss_fn
    
def main():
        #noise = tf.random.normal([1,256,256,3])
        batch_size = 2 #Normally 1000
        batch_num = 1 #Normally 100
        n = 1 #Number of
        #test_x, test_y = load_minibatch(num=5, start=0)
        Model_EncD2E = EncoderD2E()
        Model_EncP2E = EncoderP2E()
        D2E_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9)
        P2E_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9)

        #Keep track of Losses
        Encoding_losses = []
        
        for epoch in range(3000):
            train_P, train_D, y_train = load_minibatch(batch_size, start=-1, n=5)
            print("epoch: ", epoch)
            average_encoding_cost = 0
            average_D2E_cost = 0
            counter = 0
            for image in range(0,train_P.shape[0]-1,batch_size):
                print("image: ", image)
                with tf.GradientTape() as P2E_tape, tf.GradientTape() as D2E_tape:
                # Find codes for Photos
                    P_codes = Model_EncP2E(train_P[image:min(image+batch_size,train_P.shape[0]-1)],training=True)
                    # Find codes for Doodles
                    D_codes = Model_EncP2E(train_D[image:min(image+batch_size,train_D.shape[0]-1)],training=True)
                #Loss
                    encodingLoss = EmbeddingCost(P_codes, D_codes, y_train, 1, n)
                
                #Tracking loss
                    average_encoding_cost += encodingLoss
                    counter += 1

            ### Gradient Decent ###
            #P2E Encoder
                gradients_P2E = P2E_tape.gradient(encodingLoss, Model_EncP2E.trainable_variables)
                P2E_optimizer.apply_gradients(zip(gradients_P2E, Model_EncP2E.trainable_variables))
            
            #D2E Encoder
                gradients_D2E = D2E_tape.gradient(encodingLoss, Model_EncD2E.trainable_variables)
                D2E_optimizer.apply_gradients(zip(gradients_D2E, Model_EncD2E.trainable_variables))
        

        print("Embedding Cost: ", average_disc1_cost/counter)
        Encoding_losses.append(np.mean(average_encoding_cost)/counter)
        

        #np.savetxt('losses_data',np.array([gen_losses,disc1_losses,disc1_human_losses,disc1_gen_losses,disc2_losses,disc2_human_losses,disc2_gen_losses]))

        # Plot loss and save train/test images on every iteration
        #output = generator_model(test_x[test_x.shape[0]-1:test_x.shape[0]], training=False)
        #plt.imshow(output[0, :, :, :])
        #plt.savefig("test-" + str(len(gen_losses)) + ".png")
        #plt.clf()
        #output = generator_model(test_x[0:1], training=False)
        #plt.imshow(output[0, :, :, :])
        #plt.savefig("train-" + str(len(gen_losses)) + ".png")
        #plt.clf()
        #plt.plot(gen_losses)
        #plt.savefig("gen_losses.png")
        #plt.clf()
        #plt.plot(disc1_losses)
        #plt.savefig("disc1_losses.png")
        #plt.clf()
        #plt.plot(disc1_gen_losses)
        #plt.savefig("disc1_gen_losses.png")
        #plt.clf()
        #plt.plot(disc1_human_losses)
        #plt.savefig("disc1_human_losses.png")
        #plt.clf()
        #plt.plot(disc2_losses)
        #plt.savefig("disc2_losses.png")
        #plt.clf()
        #plt.plot(disc2_gen_losses)
        #plt.savefig("disc2_gen_losses.png")
        #plt.clf()
        #plt.plot(disc2_human_losses)
        #plt.savefig("disc2_human_losses.png")
        #plt.clf()

        # plt.imshow(output[0, :, :, :])
            # plt.show()

        # Save a checkpoint every other iteration
        #if epoch % 2 == 0:
        #    checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == "__main__":
    main()
