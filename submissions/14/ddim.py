#!/usr/bin/env python3
import argparse
import datetime
import os
import re
from typing import Dict
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
tf.get_logger().addFilter(lambda m: "Analyzer.lamba_check" not in m.getMessage())  # Avoid pesky warning

from image64_dataset import Image64Dataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--channels", default=32, type=int, help="CNN channels in the first stage.")
parser.add_argument("--dataset", default="oxford_flowers102", type=str, help="Image64 dataset to use.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--ema", default=0.999, type=float, help="Exponential moving average momentum.")
parser.add_argument("--epoch_images", default=50_000, type=int, help="Images per epoch.")
parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
parser.add_argument("--loss", default="MeanAbsoluteError", type=str, help="Loss object to use.")
parser.add_argument("--plot_each", default=None, type=int, help="Plot generated images every such epoch.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--sampling_steps", default=50, type=int, help="Sampling steps.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--stage_blocks", default=2, type=int, help="ResNet blocks per stage.")
parser.add_argument("--stages", default=4, type=int, help="Stages to use.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
# If you add more arguments, ReCodEx will keep them with your default values.


# The diffusion model architecture building blocks.
class SinusoidalEmbedding(tf.keras.layers.Layer):
    """Sinusoidal embeddings used to embed the current noise rate."""
    def __init__(self, dim, *args, **kwargs):
        assert dim % 2 == 0  # The `dim` needs to be even to have the same number of sin&cos.
        super().__init__(*args, **kwargs)
        self.dim = dim

    def get_config(self):
        return {"dim": self.dim}

    def call(self, inputs):
        # Compute the sinusoidal embeddings of the inputs in `[0, 1]` range.
        # The `inputs` have shape `[..., 1]`, and the embeddings should have
        # a shape `[..., self.dim]`, where for `0 <= i < self.dim/2`,
        # - the value on index `[..., i]` should be
        #     `sin(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        # - the value on index `[..., self.dim/2 + i]` should be
        #     `cos(2 * pi * inputs / 20 ** (2 * i / self.dim))`
        em_part1 = tf.sin(2 * np.pi * inputs / 20 ** (2 * tf.range(0, self.dim // 2, dtype=tf.float32) / self.dim))
        em_part2 = tf.cos(2 * np.pi * inputs / 20 ** (2 * tf.range(0, self.dim // 2, dtype=tf.float32) / self.dim))
        return tf.concat([em_part1, em_part2], axis=-1)


def ResidualBlock(inputs, width, noise_embeddings):
    """A residual block with two BN+Swish+3x3Conv, adding noise embeddings in the middle."""
    # Compute the residual connection. If the number of filters
    # in the input is the same as `width`, use unmodified `inputs`; otherwise,
    # pass it through a 1x1 convolution with `width` filters.
    if inputs.shape[-1] == width:
        residual = inputs
    else:
        residual = tf.keras.layers.Conv2D(width, 1)(inputs)

    # Pass `inputs` through a BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding and no bias (because it will be processed by a later batch normalization).
    hidden = tf.keras.layers.BatchNormalization()(inputs)
    hidden = tf.nn.swish(hidden)
    hidden = tf.keras.layers.Conv2D(width, 3, padding='same', use_bias=False)(hidden)

    # Pass `noise_embeddings` through a dense layer with `width` outputs and Swish
    # activation, and add it to `hidden`.
    hidden += tf.keras.layers.Dense(width, activation=tf.nn.swish)(noise_embeddings)

    # Pass `hidden` through another BatchNormalization, Swish activation, and 3x3 convolution
    # with "same" padding and no bias. Furthermore, initialize the kernel of the convolution to all
    # zeros, so that after initialization, the whole residual block is an identity.
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.nn.swish(hidden)
    hidden = tf.keras.layers.Conv2D(width, 3, padding="same", use_bias=False, kernel_initializer='zeros')(hidden)

    hidden += residual
    return hidden


# The DDIM model
class DDIM(tf.keras.Model):
    def __init__(self, args: argparse.Namespace, data: tf.data.Dataset) -> None:
        super().__init__()

        # Create the network inputs.
        images = tf.keras.layers.Input([Image64Dataset.H, Image64Dataset.W, Image64Dataset.C])
        noise_rates = tf.keras.layers.Input([1, 1, 1])

        # Embed noise rates using the `SinusoidalEmbedding` with `args.channels` dimensions.
        noise_embedding = SinusoidalEmbedding(args.channels)(noise_rates)

        # Process `images` using an initial 3x3 convolution with `args.channels` filters
        # and "same" padding.
        hidden = tf.keras.layers.Conv2D(args.channels, 3, padding='same')(images)

        # Downscaling stages
        outputs = []
        for i in range(args.stages):
            # For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
            # with `args.channels << i` filters and with the `noise_embedding`, and append
            # every result to the `outputs` array.
            for _ in range(args.stage_blocks):
                hidden = ResidualBlock(hidden, args.channels << i, noise_embedding)
                outputs.append(hidden)

            # Downscale `hidden` with a 3x3 convolution with stride 2,
            # `args.channels << (i + 1)` filters, and "same" padding.
            hidden = tf.keras.layers.Conv2D(args.channels << (i + 1), 3, 2, 'same')(hidden)

        # Middle block
        # For `args.stage_blocks` times, pass the `hidden` through a `ResidualBlock`
        # with `args.channels << args.stages` filters.
        for _ in range(args.stage_blocks):
            hidden = ResidualBlock(hidden, args.channels << args.stages, noise_embedding)

        # Upscaling stages
        for i in reversed(range(args.stages)):
            # Upscale `hidden` with a 4x4 transposed convolution with stride 2,
            # `args.channels << i` filters, and "same" padding.
            hidden = tf.keras.layers.Conv2DTranspose(args.channels << i, 4, 2, 'same')(hidden)

            # For `args.stage_blocks` times, concatenate `hidden` and `outputs.pop()`,
            # and pass the result through a `ResidualBlock` with `args.channels << i` filters.
            for _ in range(args.stage_blocks):
                hidden = tf.concat([hidden, outputs.pop()], axis=-1)
                hidden = ResidualBlock(hidden, args.channels << i, noise_embedding)

        # Verify that all outputs have been used.
        assert len(outputs) == 0

        # Compute the final output by passing `hidden` through a
        # BatchNormalization, Swish activation, and a 3x3 convolution with
        # `Image64Dataset.C` channels and "same" padding, with kernel of
        # the convolution initialized to all zeros.
        hidden = tf.keras.layers.BatchNormalization()(hidden)
        hidden = tf.nn.swish(hidden)
        outputs = tf.keras.layers.Conv2D(Image64Dataset.C, 3, padding='same', kernel_initializer='zeros')(hidden)

        self._network = tf.keras.Model(inputs=[images, noise_rates], outputs=outputs)

        # Create the EMA network, which will be updated by exponential moving averaging.
        self._ema_network = tf.keras.models.clone_model(self._network)

        # Create the image normalization layer and estimate the mean and variance using `data`.
        self._image_normalization = tf.keras.layers.Normalization()
        self._image_normalization.adapt(data)

        # Store required arguments for later usage.
        self._ema_momentum = args.ema
        self._seed = args.seed

        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    def _image_denormalization(self, images):
        """Invert the `self._image_normalization`, returning an image represented using bytes."""
        images = self._image_normalization.mean + images * self._image_normalization.variance**0.5
        images = tf.clip_by_value(images, 0, 255)
        images = tf.cast(images, tf.uint8)
        return images

    def _diffusion_rates(self, times):
        """Compute signal and noise rates for the given times."""
        starting_angle, final_angle = 0.2, 1.55
        # For a vector of `times` in [0, 1] range, return a pair of corresponding
        # `(signal_rates, noise_rates)`. The signal and noise rates are computed as
        # cosine and sine of an angle which is a linear interpolation from `starting_angle`
        # of 0.2 rad (for time 0) to `final_angle` of 1.55 rad (for time 1).
        # Because we use the rates as multipliers of image batches, reshape the rates
        # to a shape `[batch_size, 1, 1, 1]`, assuming `times` has a shape `[batch_size]`.
        _range = final_angle - starting_angle
        angle_range = (times * _range + starting_angle)[:, None, None, None]
        signal_rates, noise_rates = tf.cos(angle_range), tf.sin(angle_range)

        return signal_rates, noise_rates

    def train_step(self, images: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Perform a training step."""
        # Normalize the images so have on average zero mean and unit variance.
        images = self._image_normalization(images)
        # Generate a random noise of the same shape as the `images`.
        noises = tf.random.normal(tf.shape(images), seed=self._seed)
        # Generate a batch of times when to perform the loss computation in.
        times = tf.random.uniform(tf.shape(images)[:1], seed=self._seed)

        # Compute the signal and noise rates using the sampled `times`.
        signal_rates, noise_rates = self._diffusion_rates(times)

        # Compute the noisy images utilizing the computed signal and noise rates.
        noisy_images = images * signal_rates + noises * noise_rates

        with tf.GradientTape() as tape:
            # Predict the noise by the `self._network`. Do not forget to also pass
            # the `training=True` argument (to run batch normalizations in training regime).
            predicted_noises = self._network([noisy_images, noise_rates], training=True)

            # Compute loss using the `self.compiled_loss`.
            loss = self.compiled_loss(noises, predicted_noises)

        # Perform an update step.
        self.optimizer.minimize(loss, self._network.trainable_variables, tape=tape)

        # Update the `self._ema_network` using exponential moving average.
        for ema_variable, variable in zip(self._ema_network.variables, self._network.variables):
            ema_variable.assign(self._ema_momentum * ema_variable + (1 - self._ema_momentum) * variable)

        return {metric.name: metric.result() for metric in self.metrics}

    def generate(self, initial_noise, steps):
        """Sample a batch of images given the `initial_noise` using `steps` steps."""
        images = initial_noise
        # We emply a uniformly distributed sequence of times from 1 to 0. We in fact
        # create an identical batch of them, and we also make the time of the next step
        # available in the body of the cycle, because it is needed by the DDIM algorithm.
        steps = tf.linspace(tf.ones(tf.shape(initial_noise)[0]), tf.zeros(tf.shape(initial_noise)[0]), steps + 1)

        for times, next_times in zip(steps[:-1], steps[1:]):
            # Compute the signal and noise rates of the current time step.
            signal_rates, noise_rates = self._diffusion_rates(times)

            # Predict the noise by calling the `self._ema_network` with `training=False`.
            predicted_noises = self._ema_network([images, noise_rates], training=False)

            # Compute the signal and noise rates of the next time step.
            next_signal_rates, next_noise_rates = self._diffusion_rates(next_times)

            # Predict the denoised version of `images` (i.e., the $x_0$ estimate
            # in the DDIM sampling algorithm).
            denoised_images = (images - (predicted_noises * noise_rates)) / signal_rates

            # Update the `images` according to the DDIM sampling algorithm.
            images = denoised_images * next_signal_rates + predicted_noises * next_noise_rates

        # Compute the output by passing the latest `denoised_images` through
        # the `self._image_denormalization` to obtain a `tf.uint8` representation.
        images = self._image_denormalization(denoised_images)

        return images


def main(args: argparse.Namespace) -> Dict[str, float]:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        # tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the image data.
    images64 = Image64Dataset(args.dataset).train.map(lambda example: example["image"])

    # Create the model; the image data are used to initialize the image normalization layer.
    ddim = DDIM(args, images64)

    # Create the data pipeline.
    train = images64.repeat()
    train = train.shuffle(10 * args.batch_size, seed=args.seed)
    train = train.take(args.epoch_images)
    train = train.batch(args.batch_size)
    train = train.prefetch(tf.data.AUTOTUNE)

    # Class for sampling images and storing them to TensorBoard.
    class TBSampler:
        def __init__(self, columns: int, rows: int) -> None:
            self._columns = columns
            self._rows = rows
            self._noise = tf.random.normal(
                [columns * rows, Image64Dataset.H, Image64Dataset.W, Image64Dataset.C], seed=args.seed)

        def __call__(self, epoch, logs=None) -> None:
            # After the last epoch and every `args.plot_each` epoch, generate a sample to TensorBoard logs.
            if epoch + 1 == args.epochs or (epoch + 1) % (args.plot_each or args.epochs) == 0:
                images = ddim.generate(self._noise, args.sampling_steps)
                images = tf.concat([tf.concat(list(row), axis=1) for row in tf.split(images, self._rows)], axis=0)
                with ddim.tb_callback._train_writer.as_default(step=epoch):
                    tf.summary.image("images", images[tf.newaxis])
            # After the last epoch, store statistics of the generated sample for ReCodEx to evaluate.
            if epoch + 1 == args.epochs:
                logs["sample_mean"] = tf.math.reduce_mean(tf.cast(images, tf.float32))
                logs["sample_std"] = tf.math.reduce_std(tf.cast(images, tf.float32))

    # Train the model
    ddim.compile(
        optimizer=tf.optimizers.experimental.AdamW(jit_compile=False),
        loss=getattr(tf.losses, args.loss)(),
    )
    logs = ddim.fit(train, epochs=args.epochs, callbacks=[
        tf.keras.callbacks.LambdaCallback(on_epoch_end=TBSampler(16, 10)), ddim.tb_callback])

    # Return the loss and sample statistics for ReCodEx to validate.
    return {metric: values[-1] for metric, values in logs.history.items()}


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
