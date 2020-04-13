# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:06:19 2019

@author: black

# example of using upsampling in a simple generator model
#These work best for more complex GANs?
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
import numpy as np
#128 feature maps shaped 5x5 as Input
model = Sequential()
model.add(Dense(128 * 5 * 5, input_dim = 100)) 
model.add(Reshape((5,5,128)))
model.add(UpSampling2D())   #produces single 10x10 image
model.add(Conv2D(1, (3,3), padding='same'))  
model.summary()                              

gaussian = np.random.normal(size=(1,100))

# example of using transpose conv in a simple generator model

# define model
model = Sequential()
# define input shape, output enough activations for for 128 5x5 image
model.add(Dense(128 * 5 * 5, input_dim=100))
# reshape vector of activations into 128 feature maps with 5x5
model.add(Reshape((5, 5, 128)))
# double input from 128 5x5 to 1 10x10 feature map
model.add(Conv2DTranspose(1, (3,3), strides=(2,2), padding='same'))
# summarize model
model.summary()

import pandas as pd
import itertools
virus = pd.read_csv('coronavirus_sequence.txt', sep = " ", 
        header=None)
virus = virus.iloc[:,7:-1].fillna(value='9')
virus_ls = virus.values.tolist()
#virus_joined = "".join(str(x) for x in virus_ls)

virus_ls = list(itertools.chain.from_iterable(virus_ls))
virus_joined = "".join(virus_ls)
virus_joined = virus_joined.replace("9","")

import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.layers import BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

(input_train, target_train), (input_test, target_test) = mnist.load_data()

img_width, img_height = input_train.shape[1], input_train.shape[2]
batch_size = 128
no_epochs = 100
validation_split = 0.2
verbosity = 1
latent_dim = 2
num_channels = 1

# Reshape data
input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
input_test = input_test.reshape(input_test.shape[0], img_height, img_width, num_channels)
input_shape = (img_height, img_width, num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255
"""


#
#import sqlite3
#
#from sqlite3 import Error
#
#def create_connection(path):
#
#    connection = None
#
#    try:
#
#        connection = sqlite3.connect(path)
#
#        print("Connection to SQLite DB successful")
#
#    except Error as e:
#
#        print(f"The error '{e}' occurred")
##    finally:
##        if connection:
##            connection.close()
#
#    return connection
#
#connection  = create_connection(r"C:\Users\black\OneDrive\Documents\Python Scripts\pythonsqlite.db")
#
#def execute_query(connection, query):
#    cursor = connection.cursor()
#    try:
#        cursor.execute(query)
#        connection.commit()
#        print("Query executed successfully")
#    except Error as e:
#        print(f"The error '{e}' occurred")
#
#create_users_table = """
#CREATE table IF NOT EXISTS users (
#        id INTEGER PRIMARY KEY AUTOINCREMENT,
#        name TEXT NOT NULL,
#        age INTEGER,
#        gender TEXT,
#        nationality TEXT
#        );
#                     """ 
#                     
#execute_query(connection, create_users_table)
#                     
              
             


from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


#hyperparameters

batch_size = 100
original_dim = 28*28
latent_dim = 2
intermediate_dim=256
nb_epoch = 5
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim),mean=0.)
    return z_mean + K.exp(z_log_var/2)*epsilon

x  = Input(shape=(original_dim, ), name='input')
h = Dense(intermediate_dim, activation='relu', name='encoding')(x)
z_mean = Dense(latent_dim, name='mean')(h)
z_log_var = Dense(latent_dim, name='log-variance')(h)
z = Lambda(sampling, output_shape=(latent_dim, ))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name='encoder')


input_decoder = Input(shape=(latent_dim, ), name='decoder_input')
decoder_h = Dense(intermediate_dim, activation='relu', name='decoder_h')(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid', name='flat_decoded')(decoder_h)
decoder = Model(input_decoder, x_decoded, name="decoder")

output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
#vae.summary()


def vae_loss(x, x_decoded_mean, z_log_var=z_log_var, z_mean=z_mean, 
             original_dim=original_dim):
    xent_loss = original_dim * objectives.binary_crossentropy(
            x, x_decoded_mean)
    kl_loss = -0.5* K.sum(
            1+z_log_var - K.square(z_mean) - K.exp(z_log_var),
            axis=-1)
    loss = xent_loss + kl_loss
    return loss
vae.compile(optimizer='rmsprop', loss=vae_loss)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((len(x_train), 
    np.prod(x_train.shape[1:])))

x_test = x_test.reshape((len(x_test), 
    np.prod(x_test.shape[1:])))

#vae.fit(x_train, x_train, shuffle=True,
#        nb_epoch=nb_epoch, batch_size=batch_size, 
#        validation_data=(x_test, y_test), verbose=1)

from keras.layers import Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

#GANs
#------------------
img_rows, img_cols, channels = 28,28,1
img_shape = (img_rows, img_cols, channels)
Z_dim = 100  #input to Generator


def build_generator(img_shape, z_dim):
    model = Sequential()
    
    model.add(Dense(128, input_dim = z_dim))
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Dense(28*28*1, activation='tanh'))
    
    model.add(Reshape(img_shape))
    
    #print(model.summary())
    return model


def build_discriminator(img_shape):
    model = Sequential()
    
    model.add(Flatten(input_shape=img_shape))
    
    model.add(Dense(128))
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def build_gan(generator, discriminator):
    model = Sequential()
    
    model.add(generator)
    
    model.add(discriminator)
    
    return model

discriminator = build_discriminator(img_shape)
    
    
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

generator = build_generator(img_shape, Z_dim)
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam())



#GAM Training Loop

losses = []
accuracies = []
iteration_checkpoints = []

def sample_images(generator, image_grid_rows=4, image_grid_columns=4): 
 
    z = np.random.normal(0, 1, (
            image_grid_rows * image_grid_columns, Z_dim))
    

 
    gen_imgs = generator.predict(z)                                            
 
    gen_imgs = 0.5 * gen_imgs + 0.5                                            
 
    fig, axs = plt.subplots(image_grid_rows,                                  
                             image_grid_columns, 
                             figsize=(4, 4), 
                             sharey=True, 
                             sharex=True) 
 
    cnt = 0     
    for i in range(image_grid_rows):
         for j in range(image_grid_columns): 
             axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')             
             axs[i, j].axis('off')             
             cnt += 1


def train(iterations, batch_size, sample_interval):
    
    (X_train, _), (_,_) = mnist.load_data()
    
    X_train = X_train / 127.5 - 1.0
    
    X_train = np.expand_dims(X_train, axis=3)
    
    real = np.ones((batch_size,1 ))
    
    fake =np.zeros((batch_size,1 ))

    for iteration in range(iterations):
        
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        
        imgs = X_train[idx]
        
        z = np.random.normal(0, 1, (batch_size, 100))
        
        #gen_imgs = generator.predict(z)
        
        d_loss_real = discriminator.train_on_batch(imgs, real)
        
        d_loss_fake = discriminator.train_on_batch(imgs, fake)
        
        d_loss,accuracy = 0.5*np.add(d_loss_real, d_loss_fake)
        
        z = np.random.normal(0,1, (batch_size,100))
        
        #gen_imgs = generator.predict(z)
        
        g_loss = gan.train_on_batch(z, real)
        
        if (iteration + 1) % sample_interval == 0: 
 
            losses.append((d_loss, g_loss))         
            accuracies.append(100.0 * accuracy) 
            iteration_checkpoints.append(iteration + 1) 
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
                    iteration + 1, d_loss, 100.0 * accuracy, g_loss)) 

            sample_images(generator)         
        
        
iterations = 20000
batch_size = 128
sample_interval = 1000

#train(iterations, batch_size, sample_interval)
        
#CONV-GAN
from keras.layers import Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose

img_rows, img_cols, channels = 28,28,1
img_shape = (img_rows, img_cols, channels)
Z_dim = 100  #input to Generator

def build_cgenerator(img_shape, z_dim):
    model = Sequential()
    
    model.add(Dense(7*7*256, input_shape=(z_dim, )))
    
    model.add(Reshape((7,7,256)))
    
    model.add(Conv2DTranspose(128, kernel_size=3, 
            strides=2, padding='same'))
        
    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(64, kernel_size=3, strides=1,
                              padding='same'))
    
    model.add(BatchNormalization())
    
    model.add(LeakyReLU(alpha=0.01))
    
    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))

    model.add(Activation('tanh'))
    
    print(model.summary())

    return model    

#build_cgenerator(img_shape, Z_dim)
    

def build_cdiscriminator(img_shape):
    model = Sequential()