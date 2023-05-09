#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from text_classification_dataset import TextClassificationDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Evaluate model only.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
parser.add_argument("--modelnet", default=20, type=int, help="ModelNet dimension.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--weights_file", default="weights.h5", type=str, help="Name of file for saving the trained weights.")
        


def main(args: argparse.Namespace) -> None:
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

    # Load the Electra Czech small lowercased
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/eleczech-lc-small")
    eleczech = transformers.TFAutoModel.from_pretrained("ufal/eleczech-lc-small")

    # Load the data. Consider providing a `tokenizer` to the
    # constructor of the `TextClassificationDataset`.
    facebook = TextClassificationDataset("czech_facebook", tokenizer)
    
    def create_dataset(name):
        dataset: TextClassificationDataset.Dataset = getattr(facebook, name)
        labels = None
        if name != 'test':
            labels = facebook.train.label_mapping(dataset.data["labels"])
            
        def getInputIds(token: transformers.BatchEncoding):
            return token["input_ids"]
        
        def getAttentionMask(token: transformers.BatchEncoding):
            return token["attention_mask"]
        
        docs = dataset.data["documents"]
        input_ids = list(map(getInputIds, dataset.data["tokens"]))
        input_ids: tf.RaggedTensor = tf.ragged.constant(input_ids)
        input_ids = input_ids.to_tensor()
        
        attention_mask = list(map(getAttentionMask, dataset.data["tokens"]))
        attention_mask: tf.RaggedTensor = tf.ragged.constant(attention_mask)
        attention_mask = attention_mask.to_tensor()
        
        dataset = tf.data.Dataset.from_tensor_slices(((input_ids, attention_mask), labels))
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train, dev, test = create_dataset('train'), create_dataset('dev'), create_dataset('test')
    
    input_ids = tf.keras.Input(shape=[None], dtype=tf.int32)
    attention_mask = tf.keras.Input(shape=[None], dtype=tf.int32)
    x = eleczech(input_ids, attention_mask, training=True)
    x = tf.keras.layers.GlobalAveragePooling1D()(x.last_hidden_state)
    outputs = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(x)
    model = tf.keras.Model((input_ids, attention_mask), outputs)
    
    model.compile(
            optimizer=tf.optimizers.Adam(0.00005),
            loss=tf.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )
    
    model.fit(train, 
              epochs=args.epochs, 
              validation_data=dev,
              )
    

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "sentiment_analysis.txt"), "w", encoding="utf-8") as predictions_file:
        # Predict the tags on the test set.
        predictions = model.predict(test)

        label_strings = facebook.test.label_mapping.get_vocabulary()
        for sentence in predictions:
            print(label_strings[np.argmax(sentence)], file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
