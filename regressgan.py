# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf

import os
from tensorflow.keras import layers



# Helper libraries
import imageio
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from IPython import display
import PIL
import glob
from scipy import signal

InputLength = 4
Batch = 20
OutputLength = 1



"""######网络层"""

class Linear(layers.Layer):
  '''
  线性层
  输入：
  input: [Batch,length,channel] length为信号长度
  units：输出维数
  active: 激活函数： 'tanh'(默认), 'linear', 'sigmoid'
  注意：
     使用 self.add_weight的时候需要添加 name
  输出：
  线性函数 (units)
  Author: Starhou
  Email: 1029588176@qq.com
  Date: 2020.1.30
  '''
  def __init__(self, units = 1, active='linear'):
    super(Linear, self).__init__()
    self.units = units
    self.active = active
    self.usebais = True
    if self.active == 'tanh':
      self.activefun = tf.tanh
    if self.active == 'sigmoid':
      self.activefun = tf.sigmoid
    if self.active == 'linear':
      self.activefun = tf.keras.activations.linear
  def get_config(self):
    base_config = super(covT, self).get_config()
    base_config['units'] = self.units
    base_config['active'] = self.active
    base_config['usebais'] = False
    base_config['activefun'] = self.activefun
    return base_config
  def build(self, input_shape):
    self.w = self.add_weight(name='w', shape=(input_shape[-2], self.units),
                             initializer= 'random_normal',
                              trainable=True)
    if self.usebais:
      self.b = self.add_weight(name='b', shape=(self.units,),
                             initializer= 'random_normal',
                             trainable=True)    
    super(Linear, self).build(input_shape)
  def call(self, inputs):
    if self.usebais:
      out = tf.matmul(inputs[:,:,0], self.w) + self.b
    else:
      out = tf.matmul(inputs[:,:,0], self.w)
    out = self.activefun(out)
    out = tf.expand_dims(out,-1)
    return out

"""######生成器"""

class Generator(tf.keras.Model):
  def __init__(self, InputLength=1, Batch=Batch):
    super(Generator, self).__init__()
    self.start = layers.Dense(units=1,dtype='float32')
    self.linear = Linear(units=1,active='tanh')
    self.active = layers.LeakyReLU(alpha=0.2)
    self.inference_net = tf.keras.Sequential(
    [
        self.active,
        self.linear,

       ]
     )
  def call(self, inputs):
     x = self.start(inputs)
     x = self.inference_net(inputs)
     return x
generator=Generator()

"""######判别器"""

class Discriminator(tf.keras.Model):
  def __init__(self, InputLength=InputLength+1, Batch=Batch):
    super(Discriminator, self).__init__()
    self.start = layers.Conv1D(filters = 5,kernel_size = 1, padding = 'causal',input_shape=(InputLength,1))
    self.active = layers.LeakyReLU(alpha=0.2)
    self.out = layers.Dense(1, activation='tanh',dtype='float32')
    self.inference_net = tf.keras.Sequential(
      [
       self.out,
      ]
    )
  def call(self, inputs):
    x = self.inference_net(inputs)
    return x
discriminator = Discriminator()

"""######测试运行

4个已知数据 ---> 生成器 ---> 1个所求数据

[4个已知数据，1个所求数据(真实) ]---> 判别器 

[4个已知数据，1个所求数据(生成) ]---> 判别器
"""

# #测试运行  通过
noise = tf.random.normal([1,4,1])
generateECG = generator(noise)
print(generateECG.shape)
yp = tf.concat((noise,generateECG),1)
yp = discriminator(yp)
print(yp.shape)

"""######定义损失函数和优化器"""

# 定义源损失 交叉熵
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

# 判别损失 判别器要做两件事情，既要真的趋近于-1，又要假的趋近于1
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(-0.5*tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(0.5*tf.ones_like(fake_output), fake_output)
    total_loss = real_loss+fake_loss
    return total_loss
# 生成损失  生成器使得假的趋近于-1
def generator_loss(fake_output):
    fake_loss = cross_entropy(-0.5*tf.ones_like(fake_output), fake_output)
    return fake_loss

generator_optimizer = tf.keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""######定义训练"""

# 单步训练
# 注意 `tf.function` 的使用
# 该注解使函数被“编译”
@tf.function
def train_step(ECG):
  for i in range(5):
    ## 生成数据
    generatorInput = ECG[:,:4,:]
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_ECG = generator(generatorInput, training=True)
      generated_ECG = tf.concat((generatorInput,generated_ECG),1)

      real_output = discriminator(ECG, training=True)
      fake_output = discriminator(generated_ECG, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    if i==4:
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
  return gen_loss,disc_loss,

# 定义训练
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()
    num = 0
    for image_batch in dataset:
      gen_loss,disc_loss = train_step(image_batch)
      num = num+1
      if num%10==0:
        print ('generator loss {} discriminator loss {} sec'.format(gen_loss, disc_loss))
        # 继续进行时为 GIF 生成图像
        # display.clear_output(wait=True)
        generate_and_save_images(generator,
                    epoch + 1,
                    seed)
    # 每 15 个 epoch 保存一次模型
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    

  # 最后一个 epoch 结束后生成图片
  display.clear_output(wait=True)
  generate_and_save_images(generator,
              epochs,
              seed)

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    print(predictions.numpy()[-1])

"""######加载数据"""

csv_data = pd.read_csv("data/new/data13.csv", header=None)
data = np.array(csv_data)
traindata = data[1:145,:5]
testdata = data[145:433,:5]
# testdata[:,-1]=0

traindata = np.expand_dims(traindata,2)
traindata = traindata.astype(np.float32)

testdata = np.expand_dims(testdata,2)
testdata = testdata.astype(np.float32)

train_dataset = tf.data.Dataset.from_tensor_slices(traindata).shuffle(60000).batch(Batch)

# 测试数据
seed = testdata[0:5,:4,:]

"""######训练保存"""

train(train_dataset, 50)

"""#####恢复模型"""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
