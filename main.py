import opts;
from training import Trainer;
#from keras.utils import plot_model

def Main():
    opt = opts.parse();
    for split in opt.splits:
        print('+-- Split {} --+'.format(split));
        trainer = Trainer(opt, split);
        trainer.Train();
        #break;

if __name__ == '__main__':
    Main()
