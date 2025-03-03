import argparse
import math
import os
import gc

import tensorflow as tf
import ncps
from ncps.tf.ltc import LTC

import numpy as np
from tensorflow_addons.optimizers import AdamW

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from wire_neurons import WiredNeurons
import get_model
from LRCU import LRCU_Cell
from MGULayer import MGU

from tools import mse_loss_fn

def store_weights(model, filename):
    """
    Stores weights of model in a Numpy file. This function is needed instead of the built-in weight saving
    methods because layer names may be different with with TimeDistributed and non-TimeDistributed version of the model
    """
    serial = {}
    for v in model.variables:
        name = v.name
        # Remove "rnn/" from start
        if name.startswith("rnn/"):
            name = name[len("rnn/") :]
        if name in serial.keys():
            raise ValueError(f"Duplicate weight name: {name}")
        serial[name] = v.numpy()
    np.savez(filename, **serial)

class BackupToBestValEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, name, ckpt_dir='ckpt'):
        super(BackupToBestValEpochCallback, self).__init__()
        self._name = name
        self._ckpt_dir = ckpt_dir
        self._best_val_loss = math.inf
        self._train_loss_mse_when_best_val_loss = math.inf

        self._best_epoch = None
        self.copied_weights = None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_loss"] <= self._best_val_loss:
            self.copied_weights = self.model.get_weights()
            self._best_epoch = epoch
            self._best_val_loss = logs["val_loss"]
            self._train_loss_mse_when_best_val_loss = logs["loss"]
        if (epoch+1)%10 == 0:
            store_weights(self.model, f"{self._ckpt_dir}/{self._name}_epoch{epoch}.npz")
        gc.collect()
        tf.keras.backend.clear_session()


    def on_train_end(self, logs=None):
        if self.copied_weights is not None:
            print(
                f"Restoring weights to epoch {self._best_epoch} with val_loss={self._best_val_loss:0.4g} (train_loss={self._train_loss_mse_when_best_val_loss:0.4g})"
            )
            self.model.set_weights(self.copied_weights)

        filename = "ckpt/summary.txt"
        with open(filename, "a") as f:
            f.write(
                f"Model: {self._name} \nBest epoch: {self._best_epoch}, train loss mse: {self._train_loss_mse_when_best_val_loss:0.4g}, val loss: {self._best_val_loss:0.4g})\n\n"
            )
        store_weights(self.model, f"{self._ckpt_dir}/{self._name}_bestepoch.npz")
        
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="lstm")

parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--seq_len", type=int, default=25)

parser.add_argument("--lr", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=0.1)

parser.add_argument("--lr_decay", default="cosine")

parser.add_argument("--epochs", type=int, default=30)

parser.add_argument("--steps_per_epochs", type=int, default=100)

parser.add_argument("--num_neurons", type=int, default=32)

parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=256)
parser.add_argument("--channels", type=int, default=2)

parser.add_argument("--output", type=int, default=1)

parser.add_argument("--ckpt_dir", type=str, default="ckpt")

parser.add_argument("--learning_curves_dir", type=str, default="learning_curves")

parser.add_argument("--id", type=int, default=1)

args = parser.parse_args()

steps_per_epoch = args.steps_per_epochs
batch_size = args.batch_size
seq_len = args.seq_len
ckpt_dir = args.ckpt_dir
learning_curves_dir = args.learning_curves_dir
loss_fn = mse_loss_fn
input_dim = (args.height, args.width, args.channels)

alpha = args.alpha
learning_rate_cosine_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=args.lr, alpha=alpha, decay_steps=args.epochs * steps_per_epoch)
opt = AdamW(learning_rate=learning_rate_cosine_fn, weight_decay=1e-6)

name = f"{args.model}_fully{args.num_neurons}_convhead_dense64_bs_{batch_size}_seqlen_{seq_len}_lr_{args.lr}_alpha_{alpha}_epochs_{args.epochs}_id{args.id}"
    
# We'll do the input and output mapping in the setup_model function
wiring = ncps.wirings.FullyConnected(args.num_neurons)

if args.model == "ltc":
    rnn = LTC(wiring, return_sequences=True, input_mapping=None, output_mapping=None)
elif args.model == "lrc_symmetric":
    rnn = WiredNeurons(LRCU_Cell, wiring, return_sequences=True, elastance='normal_dist', input_mapping=None, output_mapping=None)
elif args.model == "lstm":
    rnn = tf.keras.layers.LSTM(args.num_neurons, implementation=1, return_sequences=True)
elif args.model == "gru":
    rnn = tf.keras.layers.GRU(args.num_neurons, return_sequences=True)
elif args.model == "mgu":
    cell = MGU(args.num_neurons)
    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)
elif args.model == "simple_rnn":
    rnn = tf.keras.layers.SimpleRNN(args.num_neurons, return_sequences=True)
else:
    raise ValueError(f"Unknown model '{args.model}'")

model = get_model.setup_model(input_dim=input_dim, output_dim=args.output, rnn = rnn)

model.compile(
    optimizer=opt,
    loss=loss_fn,
)
model.summary()

os.makedirs(learning_curves_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

data_dir = './data/extracted'

from dataloader import create_datasets, configure_for_performance
train_dataset, val_dataset, _ = create_datasets(data_dir, seq_len)

train_dataset = configure_for_performance(train_dataset, batch_size=batch_size)
val_dataset = configure_for_performance(val_dataset, batch_size=batch_size)

hist = model.fit(
    train_dataset,
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_dataset,
    validation_steps=steps_per_epoch,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"{learning_curves_dir}/{name}.csv"),
        BackupToBestValEpochCallback(name=name, ckpt_dir=ckpt_dir),
    ],
)