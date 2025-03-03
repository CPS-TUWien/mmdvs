import ncps
import tensorflow as tf
from typing import Optional, Union


@tf.keras.utils.register_keras_serializable(package="Custom", name="WiredNeurons")
class WiredNeurons(tf.keras.layers.RNN):
    def __init__(
        self,
        cell_type,
        wiring,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        input_mapping="affine",
        output_mapping="affine",
        ode_solver = None,
        elastance=None,
        epsilon=1e-8,
        initialization_ranges=None,
        return_sequences: bool = False,
        return_state: bool = False,
        go_backwards: bool = False,
        stateful: bool = False,
        unroll: bool = False,
        time_major: bool = False,
        **kwargs,
    ):

        cell = cell_type(
            wiring=wiring,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            elastance = elastance,
            epsilon=epsilon,
            initialization_ranges=initialization_ranges,
            **kwargs,
        )

        super(WiredNeurons, self).__init__(
                    cell,
                    return_sequences,
                    return_state,
                    go_backwards,
                    stateful,
                    unroll,
                    time_major,
                    **kwargs,
                )