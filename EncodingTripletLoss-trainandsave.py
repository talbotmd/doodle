import os
import numpy as np
np.random.seed(0)
import h5py
import tensorflow as tf
np.random.seed(0)
import matplotlib.pyplot as plt
from pylab import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, LeakyReLU, Dropout, Conv2DTranspose
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform,he_uniform
from keras.utils import plot_model,normalize
from keras.regularizers import l2
import random
from keras import backend as K


'''def encoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5):
    #initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential()
    model.add(Conv2D(num_filters,3,strides=1,padding='same',kernel_initializer='he_uniform',use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
        
    if apply_batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())
    if apply_dropout:
        model.add(Dropout(dropout_prob))
    return model
    
def decoder_layer(num_filters, apply_batchnorm=True,apply_dropout=False, dropout_prob=0.5):
    #initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential()
    model.add(Conv2DTranspose(num_filters,3,strides=2,padding='same',kernel_initializer='he_uniform',use_bias=False))
    
    if apply_batchnorm:
        model.add(BatchNormalization())
        model.add(LeakyReLU())
    if apply_dropout:
        model.add(Dropout(dropout_prob))
    return model

def build_EncoderP2E_dep(input_shape):
    inputs = Input(shape=input_shape)
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
    layer = Conv2D(1,3,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid') #Make fully connected
    outputs = layer(outputs)
    return Model(inputs=inputs, outputs=outputs, name = "EncoderP2E")
    
def build_EncoderD2E_dep(input_shape):
    inputs = Input(shape=input_shape)
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
    layer = Conv2D(1,3,strides=2,padding='same',kernel_initializer=tf.random_normal_initializer(0., 0.02),
        activation='sigmoid') #Make fully connected
    outputs = layer(outputs)
    return Model(inputs=inputs, outputs=outputs,  name = "EncoderD2E")
'''
    
def build_EncoderD2E(input_shape, embeddingsize=128):
    '''
    Define the neural network to learn image similarity
    Input :
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture
    '''
     # Convolutional Neural Network
    network = Sequential()
    network.add(Conv2D(32, (7,7), activation='relu',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    
    network.add(Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    return network
    
    
def build_EncoderP2E(input_shape, embeddingsize=128):
    '''
    Define the neural network to learn image similarity
    Input :
            input_shape : shape of input images
            embeddingsize : vectorsize used to encode our picture
    '''
     # Convolutional Neural Network
    network = Sequential()
    network.add(Conv2D(32, (7,7), activation='relu',
                     input_shape=input_shape,
                     kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(MaxPooling2D())
    network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
                     kernel_regularizer=l2(2e-4)))
    network.add(Flatten())
    network.add(Dense(128, activation='relu',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    
    network.add(Dense(embeddingsize, activation=None,
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer='he_uniform'))
    
    #Force the encoding to live on the d-dimentional hypershpere
    network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
    
    return network
    
def load_minibatch(num=1000, start=-1):
    #Assuming this loads a minibatch of 1000 (picture + doodles + augment pairs) * 3 (examples positive and negative, indicated by y_train)
    # Assumes the in output.hdf5 pairs are already shuffled and randomized. negative examples are created by adding a
    train_dataset = h5py.File('output.hdf5', "r")
    if start < 0:
        start = random.randint(0,18000)
    train_set_P_orig = np.array(train_dataset["image_dataset"][start:start+num],dtype='float32')
    train_set_D_orig = np.array(train_dataset["sketch_dataset"][start:start+num],dtype='float32')
    
    train_set_P_anchor = train_set_P_orig
    train_set_D_positive = train_set_D_orig
    y_train = np.ones([num,1],dtype='float32')
    
    #for i in range(1, n):
    train_set_D_negative =  np.array(train_dataset["bad_sketch_dataset"][start:start+num],dtype='float32') # your train set labels
    #train_set_P = tf.concat([train_set_P,train_set_D_positive train_set_P_orig], 0)
    #train_set_D = tf.concat([train_set_D, train_set_D_orig_false], 0)
    #y_train = tf.concat([y_train, np.zeros([num,1],dtype='float32')],axis=0)
    #print("y_train:",y_train.shape)

    triplets = np.zeros((3,train_set_P_orig.shape[0],train_set_P_orig.shape[1],train_set_P_orig.shape[2],train_set_P_orig.shape[3]))
    triplets[0][:,:,:,:] = train_set_P_anchor/255
    triplets[1][:,:,:,:] = train_set_D_positive/255
    triplets[2][:,:,:,:] = train_set_D_negative/255
    #train_set_P_anchor/255, train_set_D_positive/255, train_set_D_negative/255  
    
    return triplets

def load_testset():
    dataset = h5py.File('output.hdf5', 'r')
    test_set_P = np.array(dataset["test_images"],dtype="float32")
    test_set_D = np.array(dataset["test_images"],dtype="float32")
    return test_set_P, test_set_D

class TripletLossLayer(Layer):
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)
    
    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor-positive), axis=-1)
        n_dist = K.sum(K.square(anchor-negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)
    
    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
        
        
def build_model(input_shape,EncoderP2E, EncoderD2E, margin=20):
        anchor_P_input = Input(input_shape, name="anchor_P_input")
        positive_D_input = Input(input_shape, name="positive_D_input")
        negative_D_input = Input(input_shape, name="negative_D_input")
        
        # Generate the encodings (feature vectors) for the three images
        encoded_P_anchor = EncoderP2E(anchor_P_input)
        encoded_D_pos = EncoderD2E(positive_D_input)
        encoded_D_neg = EncoderD2E(negative_D_input)
    
        loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_P_anchor,encoded_D_pos,encoded_D_neg])
           
        # Connect the inputs with the outputs
        network_train = Model(inputs=[anchor_P_input,positive_D_input,negative_D_input],outputs=loss_layer)
        return network_train
        
def main():
        #noise = tf.random.normal([1,256,256,3])
        batch_size = 1000 #Normally 1000
        batch_num = 1 #Normally 100
        nCP = 10 #Save checkpoint every nCP epochs
        nSM = 100 #Save model every nSM checkpoints
        #n = tf.constant(1.0) #Number of negative examples per positive example
        #CHANGE: Assuing n is always 1train_set_p_orig
        margin = tf.constant(20.0) #triplet margin
        #test_x, test_y = load_minibatch(num=5, start=0)
        img_rows, img_cols, nc = 256, 256, 3
        input_shape = (img_rows, img_cols, nc)
        
        EncoderD2E = build_EncoderD2E(input_shape)
        EncoderP2E = build_EncoderP2E(input_shape)
        
        
        #EncoderD2E.save("./EncoderModel_Saves/EncoderD2E_" + str(5) + ".h5")
        #EncoderP2E.save("./EncoderModel_Saves/EncoderP2E_" + str(5) + ".h5")
        
        optimizer = Adam(0.0001, beta_1=0.9)
        
        network_train = build_model(input_shape,EncoderP2E, EncoderD2E, margin)
        
        network_train.compile(loss=None,optimizer=optimizer)
        network_train.summary()

        network_json = network_train.to_json()
        mode = 'a' if os.path.exists('checkpoints_triplet/model.json') else 'w'
        with open('checkpoints_triplet/model.json', mode) as json_file:
            json_file.write(network_json)

        #checkpoint_dir = "./checkpoints_triplets"
        #checkpoint_prefix = os.path.join(checkpoint_dir, "triplet")
        #checkpoint = tf.train.Checkpoint(optimizer=optimizer, network_train=network_train)
        #manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
        #print("checkpoints: ", manager.checkpoints)

        answer = input("Restore from checkpoint? (y/n)")
        if answer == 'y' or answer == 'yes':
            which_epoch = input("which epoch? (int)")
            network_train.load_weights("checkpoints_triplet/weights_" + str(which_epoch) + ".h5")
            print("loaded weights from disk")
            # test_set_P, test_set_D = load_testset()
            test_set_P, test_set_D, test_set_bad_D = load_minibatch(num=100, start=0)

            while True:
                which_image = input("Which test image would you like to see? (int)")
                dist_real = EncoderP2E.predict(test_set_P[int(which_image):int(which_image) + 1]) - EncoderD2E.predict(test_set_D[int(which_image):int(which_image) + 1])
                dist_fake = EncoderP2E.predict(test_set_P[int(which_image):int(which_image) + 1]) - EncoderD2E.predict(test_set_bad_D[int(which_image):int(which_image) + 1])
                print("dist_real: ", np.sum(np.square(dist_real)))
                print("dist_fake: ", np.sum(np.square(dist_fake)))
                plt.imshow(test_set_P[int(which_image)])
                plt.show()
                plt.imshow(test_set_D[int(which_image)])
                plt.show()
                plt.imshow(test_set_bad_D[int(which_image)])
                plt.show()

        elif answer == 'n' or answer == 'no':
            for iteration in range(1000):
                images_per_step = 1000
		
                #First choose positive and negative examples
                batch = load_minibatch(num=batch_size)
                for i in range(0,batch_size,images_per_step):
                    print("iteration: ", i)
                    print("input shape: ", np.array([batch[0,i:i+images_per_step,:,:,:], batch[1,i:i+images_per_step,:,:,:], batch[2,i:i+images_per_step,:,:,:]]).shape)
                    network_train.fit([batch[0,i:i+images_per_step,:,:,:], batch[1,i:i+images_per_step,:,:,:], batch[2,i:i+images_per_step,:,:,:]], epochs=5)
    
                    if iteration % nCP == 0:
                        network_train.save_weights("./checkpoints_triplet/weights_" + str(iteration) + ".h5")
                   
                    if iteration % nSM == 0:
                        EncoderD2E.save("./EncoderModel_Saves/EncoderD2E_" + str(iteration) + ".h5")
                        EncoderP2E.save("./EncoderModel_Saves/EncoderP2E_" + str(iteration) + ".h5")



if __name__ == "__main__":
    main()


