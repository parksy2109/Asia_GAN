import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

OUT_DIR = './CNN_OUT_img/'
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)
    print('complete')

img_shape = (28,28,1)
epochs = 5000
batch_size = 128
noise = 100
sample_interval = 100

(X_train , _), (X_test, _) =mnist.load_data()
print(X_train.shape)

X_train = X_train / 127.5 - 1 #data범위를 -1에서 1까지 만들기 위해
X_train = X_train.reshape(-1,28,28,1)# = reshape
print(X_train.shape)

#build generator
generator_model = Sequential()
generator_model.add(Dense(256*7*7, input_dim=noise))
generator_model.add(Reshape((7,7,256)))

generator_model.add(Conv2DTranspose(128, kernel_size=3,
                                    strides=2, padding='same')) #stride:2 = max pool size가 2x2 , 전체 사이즈가 두배로 늘어남
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(64, kernel_size=3,
                                    strides=1, padding='same')) #stride:1이므로 사이즈 그대로
generator_model.add(BatchNormalization())
generator_model.add(LeakyReLU(alpha=0.01))

generator_model.add(Conv2DTranspose(1, kernel_size=3,
                                    strides=2, padding='same'))
generator_model.add(Activation('tanh')) #tanh는 출력범위가 (-1,1) , sigmoid는 (0,1)

generator_model.summary()

#build discriminator

discriminator_model = Sequential()
discriminator_model.add(Conv2D(32, kernel_size=3,strides=2, padding='same', input_shape=img_shape))
discriminator_model.add(LeakyReLU(alpha=0.01))
discriminator_model.add(Conv2D(64, kernel_size=3,
                               strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))
#discriminator_model.add(Dropout(0.3))

discriminator_model.add(Conv2D(128, kernel_size=3,
                               strides=2, padding='same'))
#discriminator_model.add(BatchNormalization())
discriminator_model.add(LeakyReLU(alpha=0.01))
#discriminator_model.add(Dropout(0.3))

discriminator_model.add(Flatten())
discriminator_model.add(Dense(1, activation='sigmoid'))

discriminator_model.summary()

discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
discriminator_model.trainable = False

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

    z = np.random.normal(0, 1, (batch_size, noise))
    gan_hist = gan_model.train_on_batch(z, real)
    #DNN보다 CNN이 학습능력이 좋기때문에 G보다 D의 성능이 월등히 좋아진다 따라서 전체 결과에서는 서로 경쟁관계에서 학습하는 것이아닌 D의 성능이 압도적으로 높아져
    #전체적인 성능은 DNN보다 낮아진다 따라서,
    # for i in range(n): #D한번 학습할때 G는 n번 학습 되게끔
    #     z = np.random.normal(0, 1, (batch_size, noise))
    #     gan_hist = gan_model.train_on_batch(z, real)

    if itr % sample_interval == 0:
        print('{} [D loss: {}, acc.: {:.2f}%] [G loss: {}]'.format(itr,d_loss,d_acc*100, gan_hist))
        row = col = 4
        z= np.random.normal(0,1,(row*col, noise))
        fake_imgs = generator_model.predict((z))
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

