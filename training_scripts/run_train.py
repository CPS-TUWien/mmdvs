import argparse
import math
import os
import gc
import tensorflow as tf
import ncps
from ncps.tf.ltc import LTC

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow_addons.optimizers import AdamW
from dataset import LineFollowingDataset
from MGULayer import MGU
import get_model

class BackupToBestValEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, name):
        super(BackupToBestValEpochCallback, self).__init__()
        self._name = name
        self._best_val_loss = math.inf
        self._train_loss_mse_when_best_val_loss = math.inf

        self._best_epoch = None
        self.copied_weights = None

    def on_epoch_end(self, epoch, logs=None):
        # Keep track of the best validation loss
        if logs["val_loss"] <= self._best_val_loss:
            self.copied_weights = self.model.get_weights()
            self._best_epoch = epoch
            self._best_val_loss = logs["val_loss"]
            self._train_loss_mse_when_best_val_loss = logs["loss"]
        # Save weights every 10 epochs
        # if epoch%10 == 0:
        #     store_weights(self.model, f"ckpt/{name}_epoch{epoch}.npz")
        gc.collect()
        tf.keras.backend.clear_session()

    def on_train_end(self, logs=None):
        # Restore the best weights
        if self.copied_weights is not None:
            print(
                f"Restoring weights to epoch {self._best_epoch} with val_loss={self._best_val_loss:0.4g} (train_loss={self._train_loss_mse_when_best_val_loss:0.4g})"
            )
            self.model.set_weights(self.copied_weights)

        # Print the best epoch and the corresponding loss
        filename = "ckpt/summary.txt"
        with open(filename, "a") as f:
            f.write(
                f"Model: {self._name} \nBest epoch: {self._best_epoch}, train loss mse: {self._train_loss_mse_when_best_val_loss:0.4g}, val loss: {self._best_val_loss:0.4g})\n\n"
            )

# Loss function for training
def mse_loss_fn(y_true, y_pred, steering_weight = 1e4, velocity_weight = 0): 
    return steering_weight * tf.reduce_mean(tf.square(y_true[:,:] - y_pred[:,:])) 


parser = argparse.ArgumentParser()

parser.add_argument("--model", default="lstm", help="Type of model to use")

parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")

parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

parser.add_argument("--alpha", type=float, default=0.1, help="Alpha for cosine decay")

parser.add_argument("--lr_decay", default="cosine", help="Learning rate decay")

parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")

parser.add_argument("--steps_per_epoch", type=float, default=100, help="Steps per epoch")

parser.add_argument("--num_neurons", type=int, default=32, help="Number of neurons in the RNN")

parser.add_argument("--height", type=int, default=128, help="Height of the input image")

parser.add_argument("--width", type=int, default=256, help="Width of the input image")

parser.add_argument("--channels", type=int, default=2, help="Number of channels in the input image")

parser.add_argument("--output", type=int, default=1, help="Number of outputs")

parser.add_argument("--id", type=int, default=7, help="ID for the run")

args = parser.parse_args()

steps_per_epoch = args.steps_per_epoch
batch_size = args.batch_size
seq_len = args.seq_len
loss_fn = mse_loss_fn

# Learning rate schedule
alpha = args.alpha
if alpha > 0:
    learning_rate_cosine_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=args.lr, alpha=alpha, decay_steps=args.epochs * steps_per_epoch)
    opt = AdamW(learning_rate=learning_rate_cosine_fn, weight_decay=1e-6)
else:
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)
    
# Create the model
input_dim = (args.height, args.width, args.channels)

if args.model == "ltc":
    wiring = ncps.wirings.FullyConnected(args.num_neurons, output_dim = args.output)
    rnn = LTC(wiring, return_sequences=True)
elif args.model == "lstm":
    rnn = tf.keras.layers.LSTM(args.num_neurons, implementation=1, return_sequences=True)
elif args.model == "gru":
    rnn = tf.keras.layers.GRU(args.num_neurons, return_sequences=True)
elif args.model == "mgu":
    cell = MGU(args.num_neurons)
    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=True)
elif args.model == "conv_fully":
    rnn = None
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

# Name of the model
name = f"{args.model}_lr_{args.lr}_alpha_{alpha}_epochs_{args.epochs}_id{args.id}"

# Create the directories for storing the learning curves and the model weights
os.makedirs("learning_curves", exist_ok=True)
os.makedirs("ckpt", exist_ok=True)

# Dataset we use for training
data = LineFollowingDataset("data")

# This is how to use steps_per_epoch. For this use repeat() in the dataset
hist = model.fit(
    data.load_train_dataset(seq_len = seq_len, batch_size = batch_size),
    epochs=args.epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=data.load_valid_dataset(seq_len = seq_len, batch_size = batch_size),
    validation_steps=steps_per_epoch,
    callbacks=[
        tf.keras.callbacks.CSVLogger(f"learning_curves/{name}.csv"),
        BackupToBestValEpochCallback(name=name),
    ],
)

# If you don't want to define steps_per_epoch, you can do this
# hist = model.fit(
#     data.load_train_dataset(seq_len = seq_len, batch_size = batch_size),
#     epochs=args.epochs,
#     validation_data=data.load_valid_dataset(seq_len = seq_len, batch_size = batch_size),
#     callbacks=[
#         tf.keras.callbacks.CSVLogger(f"learning_curves/{name}.csv"),
#         BackupToBestValEpochCallback(name=name),
#     ],
# )

# Store the weights of the model after training
get_model.store_weights(model, f"ckpt/{name}.npz")