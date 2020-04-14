from tensorflow import keras;
from tensorflow.keras.models import Model;
import tensorflow.keras.layers as L

class EnvNet2:
    def __init__(self, input_length, n_class):
        self.input_length = input_length;
        self.conv1 = ConvBNReLU(32, (1,64), (1,2));
        self.conv2 = ConvBNReLU(64, (1,16), (1,2));
        self.conv3 = ConvBNReLU(32, (8,8));
        self.conv4 = ConvBNReLU(32, (8,8));
        self.conv5 = ConvBNReLU(64, (1,4));
        self.conv6 = ConvBNReLU(64, (1,4));
        self.conv7 = ConvBNReLU(128, (1,2));
        self.conv8 = ConvBNReLU(128, (1,2));
        self.conv9 = ConvBNReLU(256, (1,2));
        self.conv10 = ConvBNReLU(256, (1,2));
        self.fc1 = FCDN(4096);
        self.fc2 = FCDN(4096);
        self.output = FCDN(n_class, 'softmax', 0);

    def createModel(self):
        #batch, rows, columns, channels
        input = L.Input(shape=(1, self.input_length, 1));
        hl = self.conv1(input);
        #print(keras.backend.int_shape(hl));

        hl = self.conv2(hl);
        hl = L.MaxPooling2D(pool_size=(1,64), strides=(1,64))(hl);

        #swapaxes
        #hl = L.Reshape((64, 260, 1))(hl)
        hl = L.Permute((3, 2, 1))(hl)

        hl = self.conv3(hl);
        hl = self.conv4(hl);
        hl = L.MaxPooling2D(pool_size=(5,3), strides=(5,3))(hl)

        hl = self.conv5(hl);
        hl = self.conv6(hl);
        hl = L.MaxPooling2D(pool_size=(1,2), strides=(1,2))(hl)

        hl = self.conv7(hl);
        hl = self.conv8(hl);
        hl = L.MaxPooling2D(pool_size=(1,2), strides=(1,2))(hl)

        hl = self.conv9(hl);
        hl = self.conv10(hl);
        hl = L.MaxPooling2D(pool_size=(1,2), strides=(1,2))(hl)

        hl = L.Flatten()(hl);

        hl = self.fc1(hl);
        hl = self.fc2(hl);
        ol = self.output(hl);
        model = Model(inputs=input, outputs=ol);
        return model;

class ConvBNReLU:
    def __init__(self, filters, kernel_size, strides=(1,1), padding='valid', initial_w=keras.initializers.he_normal(), use_bias=False):
        self.conv = L.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=initial_w, use_bias=use_bias);

    def __call__(self, x):
        layer = self.conv(x);
        layer = L.BatchNormalization()(layer);
        layer = L.Activation('relu')(layer);
        return layer;

class FCDN:
    def __init__(self, units=50, activation='relu', dropout=0.5, initial_w=keras.initializers.lecun_normal()):
        self.fcn = L.Dense(units, kernel_initializer=initial_w);
        self.activation = L.Activation(activation);
        self.dropout = L.Dropout(rate=dropout) if dropout > 0 else None;

    def __call__(self, x):
        fc = self.fcn(x);
        fc = self.activation(fc);
        fc = self.dropout(fc) if self.dropout is not None else fc;
        return fc;
