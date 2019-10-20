import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'



def create_placeholders(n_P, n_D, n_y):
    P = tf.placeholder(tf.float32,[n_P,None], name='P') #Input Photo
    D_real = tf.placeholder(tf.float32,[n_D,None], name='D_real') #Real Doodle
    layers_dims_Hp2d = tf.placeholder(tf.float32,[None], name='layers_dims_Hp2d')
    layers_dims_Dh = tf.placeholder(tf.float32,[None], name='layers_dims_Dh')
    layers_dims_Dm = tf.placeholder(tf.float32,[None], name='layers_dims_Dm')
    #D_fake = tf.placeholder(tf.float32,[n_D,None]) #Fake Doodle generator from Hp2d
    #Y_match_real = tf.placeholder(tf.float32,[n_y,None]) #Labels
    #Y_human_real = tf.placeholder(tf.float32,[n_y,None])
    #Y_match_fake = tf.placeholder(tf.float32,[n_y,None]) #Labels
    #Y_human_fake = tf.placeholder(tf.float32,[n_y,None])

    return P, D_real, layers_dims_Hp2d, layers_dims_Hp2d, layers_dims_Dh, layers_dims_Dm #D_fake Y_match_real, Y_human_real, Y_match

def initialize_parameters(layers_dims_Hp2d, layers_dims_Dh, layers_dims_Dm):
# Per sample, inputs will be 
# Hp2d:  
    parameters_Hp2d = {}
    L = len(layers_dims_Hp2d)            # number of layers in the network
    for l in range(1, L):
        parameters_Hp2d['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l],layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Hp2d['b' + str(l)] = tf.get_variable("b"+str(l), [25,1], initializer = tf.zeros_initializer())

    parameters_Dm = {}
    L = len(layer_dims_Dm)            # number of layers in the network
    for l in range(1, L):
        parameters_Dm['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l],layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Dm['b' + str(l)] = tf.get_variable("b"+str(l), [25,1], initializer = tf.zeros_initializer())
       
    parameters_Dh = {}
    L = len(layer_dims_Dh)            # number of layers in the network
    for l in range(1, L):
        parameters_Dh['W' + str(l)] = tf.get_variable("W"+str(l), [layers_dims[l],layers_dims[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Dh['b' + str(l)] = tf.get_variable("b"+str(l), [25,1], initializer = tf.zeros_initializer())

    return parameters_Hp2d, parameters_Dh, paramters_Dm

def forward_prop(X, parameters, layers_dims):
    Z = {}
    A = {"A0": X}
    
    L = len(layers_dims)
    for l in range(1, L):
        W = parameters['W'+str(l)]
        b = parameters['b'+str(l)]
        Z['Z'+str(l)] = tf.add(tf.matmul(W,A['A'+str(l-1)],b))
        A['A'+str(l)] = tf.nn.sigmoid(Z['Z'+str(l)])
	
    A_out = A['A'+str(L)]
    return A_out, Z, A

#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits()

def Dh_loss(A_human_real, A_human_fake):
    real_loss = cross_entropy(tf.ones_like(A_human_real), A_human_real)
    fake_loss = cross_entropy(tf.zeros_like(A_human_fake), A_human_fake)
    total_loss = real_loss + fake_loss
    return total_loss

def Dm_loss(A_match_real, A_match_fake):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(A_match_real), A_match_real)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(A_match_fake), A_match_fake)
    total_loss = real_loss + fake_loss
    return total_loss

def Hp2d_loss(D_human_fake, A_match_fake):
    human_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(A_human_fake), logits = A_human_fake)
    match_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(A_match_fake), logits = A_match_fake)
    total_loss = human_loss + match_loss
    return total_loss


#Start Trying out the code
ops.reset_default_graph()
P, D_real, layers_dims_Hp2d, layers_dims_Hp2d, layers_dims_Dh, layers_dims_Dm = create_placeholders(500, 500, 50)
print ("P = " + str(P))
print ("D_real = " + str(D_real))
print ("layers_dims_Hp2d = " + str(layers_dims_Hp2d))
print ("layers_dims_Dh = " + str(layers_dims_Dh))
print ("layers_dims_Dm = " + str(layers_dims_Dm))

ops.reset_default_graph()

writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())
with tf.Session() as sess:
    #writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run())

##Run Forward Prop
#D_fake, _, _ = forward_prop(P, parameters_Hp2d, layers_dims_Hp2d)
##PD_real = concatenated version of P and D_real
##PD_fake = concatenated version of P and D_fake
#A_human_fake, _, - = forward_prop(D_fake,parameters_Dh, layers_dims_Dh)
#A_human_real, _, _ = forward_prop(D,parameters_Dh, layers_dims_Dh)
#A_match_fake, _, _ = forward_prop(PD_fake,parameters_Dm, layers_dims_Dm)
#A_match_real, _, _ = forward_prop(PD_real,parameters_Dm, layers_dims_Dm)




#cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#def Dh_loss(real_output, fake_output):
#    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#    total_loss = real_loss + fake_loss
#    return total_loss

#def Dm_loss(real_output, fake_output):
#    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
#    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
#    total_loss = real_loss + fake_loss
#    return total_loss


#def generator_loss(fake_output):
#    human_loss = 
#    match_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
#    return cross_entropy(tf.ones_like(fake_output), fake_output)
