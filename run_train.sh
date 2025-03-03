#!/bin/bash

conda run -n line_following python training_scripts/run_train.py --model lstm --lr 0.0001 --epochs 30 --id 1 --alpha 0.1 --batch_size 16 --seq_len 25
