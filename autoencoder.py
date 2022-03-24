from distutils.command.build_scripts import first_line_re
from pprint import pp
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import tensorflow as tf
import h5py as h5
import numpy as np
import random

# Simple S-Box lookup table
sbox=(
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16) 

HW = [bin(a).count("1") for a in range(256)]

# Calculate intermediate value by lookup input XOR key for a certain byte index
def intermediate_val(input, key, byte_idx=0):
    return sbox[input[byte_idx] ^ key[byte_idx]]

def intermediate_val_att(input, key, byte_idx=0):
    return sbox[input[byte_idx] ^ key]

# Calculate Hamming weight of sbox output
def get_hamming_weight(input, key, byte_idx=0):
    return HW[intermediate_val(input, key, byte_idx)]

def get_hamming_weight_att(input, key, byte_idx=0):
    return HW[intermediate_val_att(input, key, byte_idx)]

# Calculate SNR for each group
def calc_snr(x_input, y_input, amount_of_values):
    # Group input based on labels
    grouped_input = [[] for _ in range(amount_of_values)]
    for i in range(len(x_input)):
        grouped_input[y_input[i]].append(x_input[i])

    # Calculate Means and Variance of each group
    snr_means = [np.array([]) for _ in range(amount_of_values)]
    snr_var = [np.array([]) for _ in range(amount_of_values)]

    for group in range(amount_of_values):
        snr_means[group] = np.mean(grouped_input[group], axis=0)
        snr_var[group] = np.var(grouped_input[group], axis=0)

    return np.divide(np.var(snr_means, axis=0), np.mean(snr_var, axis=0))

############################
### Creating Autoencoder ###
############################
class ConvLayer(Layer):
  def __init__(self, filter, kernel, act, **kwargs):
    super().__init__()
    self.filter = filter
    self.kernel = kernel
    self.act = act
    super(ConvLayer, self).__init__(**kwargs)    
  
  def build(self, input_shape):
      self.conv = Conv1D(self.filter, self.kernel, padding='same')
      self.norm = BatchNormalization()
      self.acti = Activation(self.act)
  
  def get_config(self):
    config = super(ConvLayer, self).get_config()
    config.update({
        "filter": self.filter,
        "kernel": self.kernel,
        "act"   : self.act
    })
    return config


  def call(self, inputs):
    x = self.conv(inputs)
    x = self.norm(x)
    return self.acti(x)

class Conv2Layer(Layer):
  def __init__(self, filter, kernel, act, **kwargs):
    super().__init__()
    self.filter = filter
    self.kernel = kernel
    self.act = act
    super(Conv2Layer, self).__init__(**kwargs)    
  
  def build(self, input_shape):
      self.conv1 = Conv1D(self.filter, self.kernel, padding='same')
      self.conv2 = Conv1D(self.filter, self.kernel, padding='same')
      self.norm = BatchNormalization()
      self.acti = Activation(self.act)

  def get_config(self):
    config = super(Conv2Layer, self).get_config()
    config.update({
        "filter": self.filter,
        "kernel": self.kernel,
        "act"   : self.act
    })
    return config


  def call(self, inputs):
      x = self.conv1(inputs)
      x = self.conv2(x)
      x = self.norm(x)
      return self.acti(x)

class _Conv1DTranspose(Layer):
  def __init__(self,  filter, kernel, **kwargs):
    super().__init__()
    self.filter = filter
    self.kernel = kernel
    super(_Conv1DTranspose, self).__init__(**kwargs)

  def build(self, input_shape):
    self.first  = Lambda(lambda x: K.expand_dims(x, axis=2))
    self.conv   = Conv2DTranspose(self.filter, (self.kernel, 1), padding='same')
    self.second = Lambda(lambda x: K.squeeze(x, axis=2))
  
  def get_config(self):
    config = super(_Conv1DTranspose, self).get_config()
    config.update({
        "filter": self.filter,
        "kernel": self.kernel
    })
    return config

  def call(self, inputs):
    x = self.first(inputs)
    x = self.conv(x)
    return self.second(x)

class DeconvLayer(Layer):
  def __init__(self, filter, kernel, act, **kwargs):
      super().__init__()

      self.filter = filter
      self.kernel = kernel
      self.act = act

      super(DeconvLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.conv = _Conv1DTranspose(self.filter, self.kernel)
    self.norm = BatchNormalization()
    self.acti = Activation(self.act)

  def get_config(self):
      config = super(DeconvLayer, self).get_config()
      config.update({
          "filter": self.filter,
          "kernel": self.kernel,
          "act"   : self.act
      })
      return config

  def call(self, inputs):
    x = self.conv(inputs)
    x = self.norm(x)
    return self.acti(x)

class Deconv2Layer(Layer):
  def __init__(self, filter, kernel, act, **kwargs):
      super().__init__()

      self.filter = filter
      self.kernel = kernel
      self.act = act

      super(DeconvLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    self.conv1 = _Conv1DTranspose(self.filter, self.kernel)
    self.conv2 = _Conv1DTranspose(self.filter, self.kernel)
    self.norm = BatchNormalization()
    self.acti = Activation(self.act)

  def get_config(self):
      config = super(DeconvLayer, self).get_config()
      config.update({
          "filter": self.filter,
          "kernel": self.kernel,
          "act"   : self.act
      })
      return config

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.norm(x)
    return self.acti(x)

def create_model(latent_dim):
  encoder = Sequential([
    Conv2Layer(256, 2, 'selu'),
    AveragePooling1D(2),
    Conv2Layer(256, 2, 'selu'),
    AveragePooling1D(2),
    ConvLayer(128, 2, 'selu'),
    AveragePooling1D(2),
    ConvLayer(128, 2, 'selu'), 
    AveragePooling1D(2),
    ConvLayer(64, 2, 'selu'),
    AveragePooling1D(2),
    Flatten(),
    Dense(latent_dim, activation='selu')
  ], name='Encoder')

  decoder = Sequential([
    Dense((latent_dim * 64), activation='selu'),
    Reshape((latent_dim, 64)),
    UpSampling1D(2),
    DeconvLayer(64, 2, 'selu'),
    UpSampling1D(2),
    DeconvLayer(128, 2, 'selu'),
    UpSampling1D(2),
    DeconvLayer(128, 2, 'selu'),
    UpSampling1D(2),
    DeconvLayer(256, 2, 'selu'),
    UpSampling1D(2),
    DeconvLayer(256, 2, 'selu'),
    DeconvLayer(1, 2, 'sigmoid')
  ], name='Decoder')

  return encoder, decoder

if __name__ == "__main__":
  # Load in data
  file = h5.File('ATMEGA_AES_v1\ATM_AES_v1_fixed_key\ASCAD_data\ASCAD_databases\ATMega8515_raw_traces.h5', 'r')

  all_traces = file['traces']
  metadata = file['metadata']

  ## Metadata is build of four components
  ##  0 : Plain_text
  ##  1 : Cipher_text
  ##  2 : Key
  ##  3 : Mask

  all_ptext = np.asarray([item[0] for item in metadata])
  all_ctext =  np.asarray([item[1] for item in metadata])
  keys =  np.asarray([item[2] for item in metadata])
  masks =  np.asarray([item[3] for item in metadata])

  # Create template traces selection
  temp_indices = np.asarray(random.sample(range(all_traces.shape[0]), 10000))
  temp_indices = np.sort(temp_indices)
  temp_traces = all_traces[temp_indices]
  temp_ptext = all_ptext[temp_indices]

  # Create attack traces selection
  atk_indices = np.asarray(random.sample(range(all_traces.shape[0]), 1000))
  atk_indices = np.sort(atk_indices)
  atk_traces = all_traces[atk_indices]

  # Create labels for SNR calculation
  temp_label_int = [intermediate_val(a, keys[0], 5) for a in temp_ptext]
  temp_label_hw = [HW[a] for a in temp_label_int]

  # Calculate SNR & Select relevant input window
  snr_values = calc_snr(temp_traces, temp_label_hw, 9)
  indices_center = np.argmax(snr_values)
  indices_begin = (indices_center - 512) if indices_center > 511 else 0
  indices_end =  indices_center + 512 if indices_center > 511 else 1023
  snr_indices = range(indices_begin, indices_end)

  temp_traces = np.asarray(temp_traces[:, snr_indices])
  atk_traces = np.asarray(atk_traces[:, snr_indices])
  temp_traces = temp_traces.astype('float32')
  atk_traces = atk_traces.astype('float32')

  max_val = np.max(temp_traces)
  temp_traces = np.divide(temp_traces, max_val)

  max_val = np.max(atk_traces)
  atk_traces = np.divide(atk_traces, max_val)

  temp_traces = np.expand_dims(temp_traces, axis=2)
  atk_traces = np.expand_dims(atk_traces, axis=2)

  # Create Autoencoder
  latent_dim = 32
  with tf.device('/GPU:0'):
      encoder, decoder = create_model(latent_dim)
      autoencoder = Sequential([encoder, decoder])
      
      opt = Adam(learning_rate=1e-4)
      autoencoder.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
      # Fit & validate Autoencoder
      autoencoder.fit(temp_traces, temp_traces,
                      epochs=50,
                      shuffle=True,
                      validation_data=(atk_traces, atk_traces))
      
      autoencoder.summary()
      

  encoder.save("encoder_dim_32.h5")
