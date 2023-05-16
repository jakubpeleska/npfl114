#!/usr/bin/env python3
# 8a939b25-3475-4096-a653-2d836d1cbcad
# a0e27015-19ae-4c8f-81bd-bf628505a35a
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf
import transformers

from reading_comprehension_dataset import ReadingComprehensionDataset

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=2, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--evaluate", default=False, action="store_true", help="Do not train, only load trained weights and evaluate test set.")
parser.add_argument("--lr", default=0.00002, type=float, help="Learning rate.")
parser.add_argument("--load_weights", default="weights.h5", type=str, help="Load pre-trained weights.")
parser.add_argument("--weights_file", default="weights.h5", type=str, help="Name of file for saving the trained weights.")

class Model(tf.keras.Model):
    def __init__(self, robeczech: transformers.TFPreTrainedModel, tokenizer: transformers.PreTrainedTokenizer):
        x = tf.keras.layers.Input((None, ), dtype=tf.int32)

        mask = tf.cast(x != tokenizer.pad_token_id, tf.int32)
        y = robeczech(input_ids=x, attention_mask=mask).last_hidden_state

        sep_idx = tf.where(x == tokenizer.sep_token_id)
        sep_idx = tf.reshape(sep_idx, (-1, 2, 2))[:, 0, 1]

        y_ctx = tf.RaggedTensor.from_tensor(y, lengths=sep_idx)

        y_start = tf.keras.layers.Dense(1)(y_ctx)[..., 0]
        y_end = tf.keras.layers.Dense(1)(y_ctx)[..., 0]

        y_start = y_start.to_tensor(default_value=-1e9)
        y_end = y_end.to_tensor(default_value=-1e9)

        out = tf.stack([y_start, y_end], axis=1)
        super().__init__(inputs=x, outputs=out)
        self.tokenizer = tokenizer


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

    # Load the pre-trained RobeCzech model
    tokenizer = transformers.AutoTokenizer.from_pretrained("ufal/robeczech-base")
    robeczech = transformers.TFAutoModel.from_pretrained("ufal/robeczech-base")

    # Load the data
    reading = ReadingComprehensionDataset()
    
    def create_dataset(name, load=False):
        if load:
            dataset = tf.data.Dataset.load(f'./{name}_reading')
        else:
            dataset: ReadingComprehensionDataset.Dataset = getattr(reading, name)
            
            counter = 0 
            
            input_ids = []
            answers = [] if name != 'test' else None
            questions = [] if name == 'test' else None
            
            for p in dataset.paragraphs:
                context = p["context"]
                for q in p["qas"]:
                    counter += 1
                    print(counter, end='\r')
                    cq = context + tokenizer.sep_token + q["question"]
                    tokens = tokenizer(cq, return_offsets_mapping=True, return_special_tokens_mask=True, padding=True)
                    input_ids.append(tokens.input_ids)
                    
                    if name != 'test':
                        assert len(q["answers"]) == 1
                        a = q["answers"][0]
                        start_idx = tokens.char_to_token(a["start"])
                        end_idx = tokens.char_to_token(a["start"] + len(a["text"]) - 1)
                        answers.append((start_idx, end_idx))
                    else:
                        questions.append((context, tokens))
            
            input_ids: tf.RaggedTensor = tf.ragged.constant(input_ids)
            input_ids = input_ids.to_tensor(default_value=tokenizer.pad_token_id)
            
            print(f'postprocessing {name}')
            
            dataset = tf.data.Dataset.from_tensor_slices((input_ids, answers))
            
            dataset.save(f'./{name}_reading')
        
        dataset = dataset.shuffle(len(dataset), seed=args.seed) if name == "train" else dataset
        dataset = dataset.batch(args.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f'finished {name}')
        
        if name == 'test':
            return dataset, questions
        
        return dataset
    
    
    load = True
    train, dev = create_dataset('train', load=load), create_dataset('dev', load=load)
    
    # Create the model and train it
    model = Model(robeczech, tokenizer)
    
    weights_file = args.weights_file
    
    metrics = [tf.metrics.SparseCategoricalAccuracy(name="accuracy")]
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(weights_file, 
                                                       save_best_only=True, 
                                                       save_weights_only=True, 
                                                       monitor='val_accuracy', 
                                                       mode='max')
    callbacks = [checkpoint_cb]
    
    model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=args.lr),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=metrics,
        )
    
    if not args.evaluate:
        if args.load_weights != None:
            model.load_weights(args.load_weights, True, True)
        model.fit(train, validation_data=dev, epochs=args.epochs, callbacks=callbacks)

    model.load_weights(weights_file, True, True)
    
    test, questions = create_dataset('test')

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "reading_comprehension.txt"), "w", encoding="utf-8") as predictions_file:
        
        logits = model.predict(test)
        logits = logits.to_tensor(default_value=-1e9)
        start_logits = logits[:, 0]
        end_logits = logits[:, 1]
        
        start_idx = tf.argmax(start_logits, axis=-1)
        end_idx = tf.argmax(tf.where(tf.sequence_mask(start_idx, tf.shape(end_logits)[-1]), -1e9, end_logits), axis=-1)

        for i, tok_idx in enumerate(zip(start_idx, end_idx)):
            context: str = questions[i][0]
            encoded: transformers.BatchEncoding = questions[i][1]
            
            start = encoded.token_to_chars(tok_idx[0])
            end = encoded.token_to_chars(tok_idx[1])
            
            end = end.end if end is not None else 0
            start = start.start if start is not None else 0
            
            answer = context[start:end]
            
            print(answer, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
