import numpy as np
import tensorflow as tf

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
            elif 'elastance_dense' in name:
                param_to_find = '/'.join(name.split("/")[2:])
                timepar = True
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

def mse_loss_fn(y_true, y_pred, steering_weight = 1e4): 
    return steering_weight * tf.reduce_mean(tf.square(y_true[:,:] - y_pred[:,:]))
