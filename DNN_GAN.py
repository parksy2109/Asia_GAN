import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './OUT_img/'
img_shape = (28,28,1)
epochs = 100000
batch_size = 128
noise = 100
sample_interval = 100


(X_train , _), (X_test, _) =mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1 #data범위를 -1에서 1까지 만들기 위해
X_train = np.expand_dims(X_train, axis=3) # = reshape
print(X_train.shape)



#build generator #noise = 100개를 주면 784크기의 이미지 한장이 나옴
generator_model = Sequential()
generator_model.add(Dense(128, input_dim=noise))
generator_model.add(LeakyReLU(alpha=0.01)) #data에 음수가 포함되어있어서 사용. activation func이므로 not layer
generator_model.add(Dense(784, activation='tanh'))
generator_model.add(Reshape(img_shape))
print(generator_model.summary())


#build discriminator
discriminator_model = Sequential()
discriminator_model.add(Flatten(input_shape=img_shape))
discriminator_model.add(Dense(128))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Dense(1, activation='sigmoid'))
print(discriminator_model.summary())


discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

discriminator_model.trainable =False #학습을 안하게끔 설정

#build GAN
gan_model = Sequential()
gan_model.add(generator_model)
gan_model.add(discriminator_model)
print(gan_model.summary())

gan_model.compile(loss='binary_crossentropy', optimizer='adam')

real = np.ones((batch_size,1))
print(real)
fake = np.zeros((batch_size,1))
print(fake)


for itr in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    z = np.random.normal(0,1,(batch_size, noise)) #100개짜리 noise 128개를 random으로 생성
    fake_imgs = generator_model.predict(z)

    d_hist_real = discriminator_model.train_on_batch(real_imgs, real)
    d_hist_fake = discriminator_model.train_on_batch(fake_imgs, fake)

    d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake)
    discriminator_model.trainable = False # GAN 모델 학습시킬때는 G만 학습하고 D는 학습시키면 안됨. D는 fake image와 real image를 구분할때만 학습

    z = np.random.normal(0,1,(batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0:
        print('{} [D loss: {}, acc.: {:.2f}%] [G loss: {}]'.format(itr,d_loss,d_acc*100, gan_hist))
        row = col = 4
        z= np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict(z)
        fake_imgs = 0.5 * fake_imgs + 0.5
        _, axs = plt.subplots(row,col, figsize=(row,col),sharey=True, sharex=True)
        cnt = 0
        for i in range(row):
            for j in range(col):
                axs[i,j].imshow(fake_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        path = os.path.join(OUT_DIR, f"img-{itr + 1}")
        plt.savefig(path)
        plt.close()

