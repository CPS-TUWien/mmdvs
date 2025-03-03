import tensorflow as tf
import get_model
from MGULayer import MGU
import ncps
from ncps.tf.ltc import LTC
import argparse
import numpy as np
import os
import csv
from LRCU import LRCU_Cell
from wire_neurons import WiredNeurons
from dataloader import create_datasets, configure_for_performance
import matplotlib.pyplot as plt
from tools import restore_weights, mse_loss_fn

parser = argparse.ArgumentParser()

parser.add_argument("--model", default="lstm")

parser.add_argument("--weight_filename", default="lstm_fully32_convhead_dense64_bs_2_seqlen_16_lr_0.0001_alpha_0.1_epochs_10_id1_bestepoch")

parser.add_argument("--seq_len", type=int, default=1, help="Sequence length")

parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--num_neurons", type=int, default=32)

parser.add_argument("--height", type=int, default=128)

parser.add_argument("--width", type=int, default=256)

parser.add_argument("--channels", type=int, default=2)

parser.add_argument("--output", type=int, default=1)

parser.add_argument("--stateful", default=False, action="store_true", help="If consecutive sequences are related")

parser.add_argument("--ckpt_dir", type=str, default="ckpt")

parser.add_argument("--save_predictions", default=False, action="store_true", help="Save predictions to file or calculate the loss only")

parser.add_argument("--bestepoch", default=False, action="store_true")

parser.add_argument("--steps", type=int, default=1000)


args = parser.parse_args()

data_dir = './data/extracted'
ckpt_dir = args.ckpt_dir
batch_size = args.batch_size
seq_len = args.seq_len
input_dim = (args.height, args.width, args.channels)
wiring = ncps.wirings.FullyConnected(args.num_neurons)

build_fns = {
    "lrc_symmetric": get_model.setup_model(input_dim, args.output, WiredNeurons(LRCU_Cell, wiring, return_sequences=True, stateful=args.stateful, elastance='normal_dist', input_mapping=None, output_mapping=None), stateful=args.stateful),
    "ltc": get_model.setup_model(input_dim, args.output, LTC(ncps.wirings.FullyConnected(args.num_neurons), return_sequences=True, stateful=args.stateful, input_mapping=None, output_mapping=None), stateful=args.stateful),
    "lstm": get_model.setup_model(input_dim, args.output, tf.keras.layers.LSTM(args.num_neurons, implementation=1, return_sequences=True, stateful=args.stateful, return_state=False), stateful=args.stateful),
    "gru": get_model.setup_model(input_dim, args.output, tf.keras.layers.GRU(args.num_neurons, return_sequences=True, stateful=args.stateful, return_state=False), stateful=args.stateful),
    "simple_rnn": get_model.setup_model(input_dim, args.output, tf.keras.layers.SimpleRNN(args.num_neurons, return_sequences=True, stateful=args.stateful, return_state=False), stateful=args.stateful),
    "mgu": get_model.setup_model(input_dim, args.output, tf.keras.layers.RNN(MGU(args.num_neurons), time_major=False, return_sequences=True, stateful=args.stateful, return_state=False), stateful=args.stateful),
}

model = build_fns[args.model]
input_shape = (None, args.height, args.width, args.channels)
model.build(input_shape)
model.summary()

model.compile(loss=mse_loss_fn)

restore_weights(model, f"./{ckpt_dir}/{args.weight_filename}.npz")

_, _, test_dataset = create_datasets(data_dir, seq_len, with_filenames=args.save_predictions)
test_dataset = configure_for_performance(test_dataset, batch_size=batch_size)

print("Predicting...")

if args.save_predictions:

    os.makedirs("predictions/test", exist_ok=True)

    targets, preds = [], []

    with open(f'predictions/test/{args.weight_filename}_stateful_{args.stateful}.csv','w') as file:
        csvwriter = csv.writer(file)
        for j, data_sample in enumerate(test_dataset):
            data, target, filename = data_sample
            predictions = model.predict(data)
            for i in range(predictions.shape[1]):
                csvwriter.writerow([str(filename[0,i].numpy()).split("/")[-1][:-1], target[0,i].numpy(), predictions[0,i,0]])
                targets.append(target[0,i].numpy())
                preds.append(predictions[0,i,0])
            if args.stateful and j > args.steps:
                break
            if not args.stateful and j>100:
                break

    print("Done")

    plt.plot(targets, label='target')
    plt.plot(preds, label='prediction')
    plt.legend()
    plt.savefig(f'predictions/test/{args.weight_filename}_stateful_{args.stateful}.png')
else:
    test_loss = model.evaluate(test_dataset, steps=args.steps)

    test_dir = f"test_loss_{args.steps}/stateful_{args.stateful}_bestepoch_{args.bestepoch}"
    os.makedirs(test_dir, exist_ok=True)

    print('mse:', test_loss)
    with open(f"{test_dir}/{args.weight_filename}.txt", "w") as f:

        f.write(f"test mse: {test_loss}")
        f.close()