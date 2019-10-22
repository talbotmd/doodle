import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

layers_dims_Hp2d = [10, 9, 8, 7, 6, 5, 5, 6, 7, 8, 9, 10]
layers_dims_Dh = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
layers_dims_Dm = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]

A = [[1, 3], [2, 6], [3, 9], [4, 12], [5, 15], [6, 18], [7, 21], [8, 24], [9, 27], [10, 30]]
B = [[1, 1], [2, 3], [5, 8], [13, 21], [34, 55], [89, 13], [21, 34], [55, 89], [5, 8], [13, 21]]

learning_rate = 1
m = len(A[0])
k = 10 #Number of interations of dicriminator training before training generator
#L_dim_H = [10, 9, 8, 7, 6, 5, 5, 6, 7, 8, 9, 10]
#L_dim_Dh = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
#L_dim_Dm = [20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 1]

#Inputs = {'P': X, 'D_real': Y}
#print("Inputs: " + str(Inputs))


def create_placeholders(n_P, n_D):
    P = tf.placeholder(tf.float32,[n_P,None], name='P') #Input Photo
    D_real = tf.placeholder(tf.float32,[n_D,None], name='D_real') #Real Doodle
    #D_fake = tf.placeholder(tf.float32,[n_D,None]) #Fake Doodle generator from Hp2d
    #Y_match_real = tf.placeholder(tf.float32,[n_y,None]) #Labels
    #Y_human_real = tf.placeholder(tf.float32,[n_y,None])
    #Y_match_fake = tf.placeholder(tf.float32,[n_y,None]) #Labels
    #Y_human_fake = tf.placeholder(tf.float32,[n_y,None])

    return P, D_real #D_fake Y_match_real, Y_human_real, Y_match

def initialize_parameters(layers_dims_Hp2d, layers_dims_Dh, layers_dims_Dm):
# Per sample, inputs will be 
# Hp2d:  
    parameters_Hp2d = {}
    L = len(layers_dims_Hp2d)            # number of layers in the network
    for l in range(1, L):
        parameters_Hp2d['WHp2d' + str(l)] = tf.get_variable("WHp2d"+str(l), [layers_dims_Hp2d[l],layers_dims_Hp2d[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Hp2d['bHp2d' + str(l)] = tf.get_variable("bHp2d"+str(l), [layers_dims_Hp2d[l],1], initializer = tf.zeros_initializer())

    parameters_Dm = {}
    L = len(layers_dims_Dm)            # number of layers in the network
    for l in range(1, L):
        parameters_Dm['WDm' + str(l)] = tf.get_variable("WDm"+str(l), [layers_dims_Dm[l],layers_dims_Dm[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Dm['bDm' + str(l)] = tf.get_variable("bDm"+str(l), [layers_dims_Dm[l],1], initializer = tf.zeros_initializer())
       
    parameters_Dh = {}
    L = len(layers_dims_Dh)            # number of layers in the network
    for l in range(1, L):
        parameters_Dh['WDh' + str(l)] = tf.get_variable("WDh"+str(l), [layers_dims_Dh[l],layers_dims_Dh[l-1]], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        parameters_Dh['bDh' + str(l)] = tf.get_variable("bDh"+str(l), [layers_dims_Dh[l],1], initializer = tf.zeros_initializer())

    return parameters_Hp2d, parameters_Dh, parameters_Dm

def forward_prop(X, parameters, layers_dims, namestring):
    Z = {}
    A = {"A"+namestring+str(0): X}
    
    L = len(layers_dims)
    for l in range(1, L):
        W = parameters['W'+namestring+str(l)]
        b = parameters['b'+namestring+str(l)]
        Z['Z'+namestring+str(l)] = tf.add(tf.matmul(W,A['A'+namestring+str(l-1)]),b)
        A['A'+namestring+str(l)] = tf.nn.sigmoid(Z['Z'+namestring+str(l)])
	
    #print(str(l))
    A_out = A['A'+namestring+str(l)]
    return A_out, Z, A

#cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits()

def Dh_loss(A_human_real, A_human_fake):
    real_loss = cross_entropy(labels = tf.ones_like(A_human_real), logits = A_human_real)
    fake_loss = cross_entropy(labels = tf.zeros_like(A_human_fake), logits = A_human_fake)
    total_loss = real_loss + fake_loss
    return total_loss

def Dm_loss(A_match_real, A_match_fake):
    real_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(A_match_real), logits = A_match_real)
    fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(A_match_fake), logits = A_match_fake)
    total_loss = real_loss + fake_loss
    return total_loss

def Hp2d_loss(A_human_fake, A_match_fake):
    human_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(A_human_fake), logits = A_human_fake)
    match_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(A_match_fake), logits = A_match_fake)
    total_loss = human_loss + match_loss
    return total_loss


#Start Trying out the code
ops.reset_default_graph() #reset default graph
P, D_real = create_placeholders(10, 10)
print ("P = " + str(P))
print ("D_real = " + str(D_real))
print ("layers_dims_Hp2d = " + str(layers_dims_Hp2d))
print ("layers_dims_Dh = " + str(layers_dims_Dh))
print ("layers_dims_Dm = " + str(layers_dims_Dm))

#Initilize Data Inputs Dictionary

with tf.Session() as sess:
    parameters_Hp2d, parameters_Dh, parameters_Dm = initialize_parameters(layers_dims_Hp2d, layers_dims_Dh, layers_dims_Dm)
    print("parameters_Hp2d " + str(parameters_Hp2d))
    print("parameters_Dm " + str(parameters_Dm))
    print("parameters_Dh " + str(parameters_Dh))


D_fake, _, _ = forward_prop(P, parameters_Hp2d, layers_dims_Hp2d, "Hp2d")
##PD_real = concatenated version of P and D_real
##PD_fake = concatenated version of P and D_fake
PD_real = tf.concat([P, D_real], 0, name="PD_real")
PD_fake = tf.concat([P, D_fake], 0, name="PD_fake")
A_human_fake, _, _ = forward_prop(D_fake,parameters_Dh, layers_dims_Dh, "Dh")
A_human_real, _, _ = forward_prop(D_real,parameters_Dh, layers_dims_Dh, "Dh")
A_match_fake, _, _ = forward_prop(PD_fake,parameters_Dm, layers_dims_Dm, "Dm")
A_match_real, _, _ = forward_prop(PD_real,parameters_Dm, layers_dims_Dm, "Dm")

# feed_dict = {x: 3}

Loss_Hp2d = Hp2d_loss(A_human_fake, A_match_fake)
Loss_Dm = Dm_loss(A_match_real, A_match_fake)
Loss_Dh = Hp2d_loss(A_human_fake, A_match_fake)

optimizer_Hp2d = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Loss_Hp2d, var_list = parameters_Hp2d)
optimizer_Dm = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Loss_Dm, var_list = parameters_Dm)
optimizer_Dh = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(Loss_Dh, var_list = parameters_Dh)

init = tf.global_variables_initializer()

#to create tensorboard graph
writer = tf.summary.FileWriter('./graphs',tf.get_default_graph())

J_Hp2d = []
J_Dh = []
J_Dm = []

for i in range(1,m):
#i = m
    with tf.Session() as sess:
        sess.run(init)
        #writer = tf.summary.FileWriter('./graphs', sess.graph)
        Loss_Hp2d, Loss_Dm, Loss_Dh, = sess.run(Loss_Hp2d, feed_dict = {P: A, D_real: B}), sess.run(Loss_Dm, feed_dict = {P: A, D_real: B}), sess.run(Loss_Dh, feed_dict = {P: A, D_real: B})
        _, _ = sess.run(optimizer_Dm, feed_dict = {P: A, D_real: B}), sess.run(optimizer_Dh, feed_dict = {P: A, D_real: B})
        if np.mod(i,k) == 0:
            _ = sess.run(optimizer_Hp2d, feed_dict = {P: A, D_real: B})
        
J_Hp2d.append(Loss_Hp2d)
J_Dh.append(Loss_Dh)
J_Dm.append(Loss_Dm)
            
print("Loss in Hp2d = "+str(Loss_Hp2d))
print("Loss in Dh = "+str(Loss_Dh))
print("Loss in Dm = "+str(Loss_Dm))
sess.close()





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
