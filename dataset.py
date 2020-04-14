import os;
import numpy as np;
import random;
import utils as U;
from tensorflow import keras;

class Generator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, samples, labels, options, train=True):
        'Initialization'
        self.data = [(samples[i], labels[i]) for i in range (0, len(samples))];
        self.opt = options;
        self.train = train;
        self.batch_size = options.batchSize if train else options.batchSize // options.nCrops;
        self.mix = (options.BC and train);
        self.preprocess_funcs = self.preprocess_setup();
        #self.__getitem__(3);

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data) / self.batch_size));
        #return len(self.samples);

    def __getitem__(self, batchIndex):
        'Generate one batch of data'
        batchX, batchY = self.generate_batch(batchIndex);
        batchX = np.expand_dims(batchX, axis=1)
        batchX = np.expand_dims(batchX, axis=3)
        #print(batchX.shape);
        #exit();
        return batchX, batchY

    def generate_batch(self, batchIndex):
        'Generates data containing batch_size samples'
        sounds = [];
        labels = [];
        indexes = None;
        for i in range(self.batch_size):
            # Generate indexes of the batch
            if self.mix:  # Training phase of BC learning

                # Select two training examples
                while True:
                    sound1, label1 = self.data[random.randint(0, len(self.data) - 1)]
                    sound2, label2 = self.data[random.randint(0, len(self.data) - 1)]
                    if label1 != label2:
                        break
                sound1 = self.preprocess(sound1)
                sound2 = self.preprocess(sound2)

                # Mix two examples
                r = np.array(random.random())
                sound = U.mix(sound1, sound2, r, self.opt.fs).astype(np.float32)
                eye = np.eye(self.opt.nClasses)
                label = (eye[label1] * r + eye[label2] * (1 - r)).astype(np.float32)

            else:  # Training phase of standard learning or testing phase
                #print(batchIndex);
                if indexes == None:
                    indexes = self.data[batchIndex*self.batch_size:(batchIndex+1)*self.batch_size];
                else:
                    if i >= len(indexes):
                        break;

                sound, target = indexes[i];
                sound = self.preprocess(sound).astype(np.float32)
                #label = (np.eye(self.opt.nClasses))[int(label)]
                label = np.zeros((self.opt.nCrops, self.opt.nClasses));
                label[:,target] = 1;

            if self.train and self.opt.strongAugment:
                sound = U.random_gain(6)(sound).astype(np.float32)

            sounds.append(sound);
            labels.append(label);

        sounds = np.asarray(sounds);
        labels = np.asarray(labels);
        if not self.train:
            sounds = sounds.reshape(sounds.shape[0]*sounds.shape[1], sounds.shape[2]);
            labels = labels.reshape(labels.shape[0]*labels.shape[1], labels.shape[2]);

        return sounds, labels;

    '''
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    '''


    def preprocess_setup(self):
        if self.train:
            funcs = []
            if self.opt.strongAugment:
                funcs += [U.random_scale(1.25)]

            funcs += [U.padding(self.opt.inputLength // 2),
                      U.random_crop(self.opt.inputLength),
                      U.normalize(32768.0),
                      ]
        else:
            funcs = [U.padding(self.opt.inputLength // 2),
                     U.normalize(32768.0),
                     U.multi_crop(self.opt.inputLength, self.opt.nCrops),
                     ]
        return funcs

    def preprocess(self, sound):
        for f in self.preprocess_funcs:
            sound = f(sound)

        return sound;


def setup(opt, split):
    dataset = np.load(os.path.join(opt.data, opt.dataset, 'wav{}.npz'.format(opt.fs // 1000)))
    # Split to train and val
    train_sounds = []
    train_labels = []
    val_sounds = []
    val_labels = []
    for i in range(1, opt.nFolds + 1):
        sounds = dataset['fold{}'.format(i)].item()['sounds']
        labels = dataset['fold{}'.format(i)].item()['labels']
        if i == split:
            val_sounds.extend(sounds)
            val_labels.extend(labels)
        else:
            train_sounds.extend(sounds)
            train_labels.extend(labels)

    # Iterator setup
    train_data = Generator(train_sounds, train_labels, opt, train=True)
    val_data = Generator(val_sounds, val_labels, opt, train=False)    

    return train_data, val_data
