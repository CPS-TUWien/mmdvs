import tensorflow as tf
import get_model
from MGULayer import MGU
import ncps
from ncps.tf.ltc import LTC
import argparse
import numpy as np
from dataset import LineFollowingDataset
import os
import csv

# Loss function for training
def mse_loss_fn(y_true, y_pred, steering_weight = 1e4, velocity_weight = 0): 
    return steering_weight * tf.reduce_mean(tf.square(y_true[:,:] - y_pred[:,:])) 

def restore_weights(model, filename, cfc=False):
    """
    Loads weights of model from a Numpy file. This function is needed instead of the built-in weight saving/loading
    methods because layer names may be different with with TimeDistributed and non-TimeDistributed version of the model
    """
    arr = np.load(filename)

    for v in model.variables:
        name = v.name
        timepar = False
        if name.startswith("wired_neurons") or name.startswith("rnn") or name.startswith("lstm") or name.startswith("gru"):
            if cfc:
                param_to_find = name.split("/", 1)[1]
            else:
                first_string = name.split("/")[0]
                second_string = name.split("/")[1]
                param_to_find = name.split("/")[2]
                timepar=False

            ok = False
            for param in arr.files:
                if '/' + param_to_find in param:
                    if timepar:
                        v.assign(arr[param])
                        ok = True
                        break
                    else:
                        if param.startswith(first_string[:3]) or param.startswith(second_string[:3]):
                            v.assign(arr[param])
                            ok = True
                            break
            if not ok:
                raise ValueError(f"var {name} not found in file '{filename}'")
        else:
            if not name in arr.files:
                raise ValueError(f"var {name} not found in file '{filename}'")
            v.assign(arr[name])

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="lstm")

parser.add_argument("--weight_filename", default="lstm_fully32_convhead_dense64_lr_0.001_alpha_0.1__epochs_50_tenthdata_id1")

parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")

parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--lr", type=float, default=0.001)

parser.add_argument("--alpha", type=float, default=0.1)

parser.add_argument("--lr_decay", default="cosine")

parser.add_argument("--epochs", type=int, default=100)

parser.add_argument("--num_neurons", type=int, default=32)

parser.add_argument("--height", type=int, default=128)

parser.add_argument("--width", type=int, default=256)

parser.add_argument("--channels", type=int, default=2)

parser.add_argument("--output", type=int, default=1)

parser.add_argument("--id", type=int, default=1)

parser.add_argument("--stateful", default=False, action="store_true", help="If consecutive sequences are related")

args = parser.parse_args()

input_dim = (args.height, args.width, args.channels)

build_fns = {
    "ltc": get_model.setup_model(input_dim, args.output, LTC(ncps.wirings.FullyConnected(args.num_neurons, output_dim = args.output), return_sequences=True), stateful=args.stateful),
    "lstm": get_model.setup_model(input_dim, args.output, tf.keras.layers.LSTM(args.num_neurons, implementation=1, return_sequences=True, return_state=False), stateful=args.stateful),
    "gru": get_model.setup_model(input_dim, args.output, tf.keras.layers.GRU(args.num_neurons, return_sequences=True, return_state=False), stateful=args.stateful),
    "simple_rnn": get_model.setup_model(input_dim, args.output, tf.keras.layers.SimpleRNN(args.num_neurons, return_sequences=True, return_state=False), stateful=args.stateful),
    "mgu": get_model.setup_model(input_dim, args.output, tf.keras.layers.RNN(MGU(args.num_neurons), time_major=False, return_sequences=True, return_state=False), stateful=args.stateful),
    "conv_fully": get_model.setup_model(input_dim, args.output, None),
}

model = build_fns[args.model]
input_shape = (None, args.height, args.width, args.channels)
model.build(input_shape)
model.summary()

model.compile(
    loss=mse_loss_fn,
)

restore_weights(model, f"./ckpt/{args.weight_filename}.npz")
data = LineFollowingDataset("data")

test_ds = data.load_test_dataset(seq_len=args.seq_len, batch_size=args.batch_size, include_filenames=True)

os.makedirs("predictions/test", exist_ok=True)

targets, preds = [], []
with open(f'predictions/test/{args.weight_filename}.csv','w') as file:
    csvwriter = csv.writer(file)
    dataloader_iterator = iter(test_ds)
    for i in range(500):
        try:
            data, target, filename = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(test_ds)
            data, target, filename = next(dataloader_iterator)
        predictions = model.predict(data)

        for i in range(predictions.shape[1]): #(batch,len)
            csvwriter.writerow([str(filename[0,i].numpy()).split("/")[-1][:-1], predictions[0,i,0]])
            targets.append(target[0,i].numpy())
            preds.append(predictions[0,i,0])

file.close()