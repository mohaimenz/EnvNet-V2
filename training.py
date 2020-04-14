import sys;
import os;
import utils as U;
from tensorflow import keras;
import models;
import dataset;
import math;
import numpy as np;
import time;
#from keras.utils import plot_model

class Trainer:
    def __init__(self, opt=None, split=0):
        self.opt = opt;
        self.split = split;

    def Train(self):
        envnet2 = models.EnvNet2(66650, 50);
        model = envnet2.createModel();
        print(model.summary());
        exit();
        trainGen, valGen = dataset.setup(self.opt, self.split);
        loss = 'kullback_leibler_divergence'
        optimizer = keras.optimizers.SGD(lr=self.opt.LR, decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True)

        model.compile(loss=loss, optimizer=optimizer , metrics=['accuracy']);

        # learning schedule callback
        lrate = keras.callbacks.LearningRateScheduler(self.GetLR);
        #best_model = keras.callbacks.ModelCheckpoint('best_model_fold-'+str(self.split )+'_epoch-{epoch:02d}_val_acc-{val_acc:.2f}.hdf5', monitor='val_acc', save_best_only=True, verbose=0);
        best_model = keras.callbacks.ModelCheckpoint('Split-'+str(self.split )+'_best_model.hdf5', monitor='val_acc', save_best_only=True, verbose=0);
        #early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=100);
        csv_logger = keras.callbacks.CSVLogger('aug-fold-'+str(self.split)+'-training.log');
        custom_evaluator = CustomCallback(self.opt, trainGen, valGen);
        #callbacks_list = [lrate, custom_evaluator, best_model, early_stopping, csv_logger];
        callbacks_list = [lrate, custom_evaluator, best_model, csv_logger];
        #callbacks_list = [lrate, custom_evaluator];

        #My custom data generator
        #model.fit_generator(trainGen, epochs=opt.nEpochs, steps_per_epoch=len(trainGen.data)//trainGen.batch_size, validation_data=valGen, validation_steps=math.ceil(len(valGen.data) / valGen.batch_size), callbacks=callbacks_list, verbose=1);
        model.fit_generator(trainGen, epochs=self.opt.nEpochs, steps_per_epoch=len(trainGen.data)//trainGen.batch_size, callbacks=callbacks_list, verbose=0);
        #model.fit_generator(trainGen, epochs=self.opt.nEpochs, steps_per_epoch=1, callbacks=callbacks_list, verbose=0);

        #print(model.summary());
    def GetLR(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.LR * np.power(0.1, decay);

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, opt, trainGen, valGen):
        self.opt = opt;
        self.train_gen = trainGen;
        self.val_gen = valGen;
        self.curEpoch = 0;
        self.curLr = opt.LR;
        self.start_time = time.time();
        self.cur_epoch_start_time = time.time();
        self.i = 'None';

    def on_train_batch_begin(self, batch, logs=None):
        '''
        print('Training: batch {} begins at {}'.format(batch, time.time()));
        '''
    def on_train_batch_end(self, batch, logs=None):
        elapsed_time = time.time() - self.start_time;
        nTrain_batches = (len(self.train_gen.data) - 1) // self.opt.batchSize + 1;
        progress = (nTrain_batches * (self.curEpoch - 1) + batch + 1) * 1.0 / (nTrain_batches * self.opt.nEpochs);
        eta = elapsed_time / progress - elapsed_time;
        line = '* Epoch: {}/{} ({}/{}) | Train: LR {} | Time: {} (ETA: {})'.format(
            self.curEpoch, self.opt.nEpochs, batch+1, nTrain_batches, self.curLr, U.to_hms(elapsed_time), U.to_hms(eta));
        sys.stderr.write('\r\033[K');
        sys.stdout.write(line);
        sys.stdout.flush();

    def on_epoch_begin(self, epoch, logs=None):
        self.curEpoch = epoch+1;
        self.curLr = Trainer(self.opt).GetLR(epoch+1);
        self.cur_epoch_start_time = time.time();

    def on_epoch_end(self, epoch, logs=None):
        val_acc, val_loss = self.validate(self.model);
        logs['val_acc'] = val_acc;
        logs['val_loss'] = val_loss;
        time_taken = time.time() - self.cur_epoch_start_time;
        sys.stderr.write('\r\033[K')
        sys.stdout.write(
            'Epoch: {}/{} | Time: {} | Train: LR {}  Loss {:.3f}%  Acc {:.3f}% | Val: Loss {:.3f}%  Acc(top1) {:.3f}%\n'.format(
                epoch+1, self.opt.nEpochs, U.to_hms(time_taken), self.curLr, logs['loss'], logs['acc'], val_loss, val_acc));
        sys.stdout.flush();

    def validate(self, model):
        y_pred = None;
        y_target = None;
        for batchIndex in range(math.ceil(len(self.val_gen.data) / self.val_gen.batch_size)):
            testX, testY = self.val_gen.__getitem__(batchIndex);
            scores = model.predict(testX, batch_size=len(testX), verbose=0);
            y_pred = scores if y_pred is None else np.concatenate((y_pred, scores));
            y_target = testY if y_target is None else np.concatenate((y_target, testY));
            #break;

        acc, loss = self.compute_accuracy(y_pred, y_target);
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target):
        #Reshape y_pred to shape it like each sample comtains 10 samples.
        y_pred = y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1]);

        #Calculate the average of class predictions for 10 crops of a sample
        y_pred = np.mean(y_pred, axis=1);

        #Get the indices that has highest average value for each sample
        y_pred = y_pred.argmax(axis=1);

        #Doing the samething for y_target
        y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(axis=1).argmax(axis=1);

        accuracy = (y_pred==y_target).mean()*100;
        loss = np.mean(y_target - y_pred);
        return accuracy, loss;
