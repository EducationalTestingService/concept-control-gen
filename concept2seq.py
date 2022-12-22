'''
concept2seq.py

Framework for training/predicting with concept2seq model, based on huggingface codebase
'''

import sys
import argparse
import csv
import os
import random
import json
import jsonlines
import string

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    BartForConditionalGeneration, BartTokenizer, 
    Seq2SeqTrainingArguments, Seq2SeqTrainer, BartConfig, BartTokenizerFast, PretrainedConfig, BartModel
  )

import torch
from torch.utils.data import random_split
from datasets import Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOKENIZER = BartTokenizerFast.from_pretrained("facebook/bart-large", max_length=64)

random.seed(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--output_path", type=str, required=False)
    parser.add_argument("--test_path", type=str, required=False)
    parser.add_argument("--mode", type=str, required=False, default=False)
    parser.add_argument("--n", type=int, required=False, default=0)
    parser.add_argument("--epochs", type=int, required=False, default=5)
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--config", type=str, required=False)
    parser.add_argument("--num_sentences", type=int, required=False, default=1)
    parser.add_argument("--extras", type=str, required=False, default=None)
    parser.add_argument("--max_concepts", type=int, required=False, default=10)
    parser.add_argument("--min_concepts", type=int, required=False, default=1)

    return parser.parse_args()

def stitch_srl(concepts:list, parse:dict) -> str:
    ''' Functionality for combining SRL parse into the input concepts for concept2seq '''
    words = parse["words"]
    concept_replacements = {}
    verbs = []

    for verb in parse["verbs"]:
        verbs.append(verb['verb'])
        tags = verb["tags"]
        for e, w in enumerate(words):
            if w in concepts and tags[e] != "O":
                concept_replacements[w] = w + "-" + (tags[e] if "-" not in tags[e] else tags[e].split("-")[1])
    new_concepts = [conc if conc not in concept_replacements else concept_replacements[conc] for conc in concepts]
    return " ".join(new_concepts)


def load_data(path:str, extras:str=None, min_concepts:int=0, max_concepts:int=10) -> list:
    ''' Load data, either for training or testing. Currently using jsonlines files. Combines "extras" with original concept inputs '''
    data = []
    data_json = jsonlines.open(path)
    for line in data_json:
        source = line["source"].lower()
        source = source.translate(str.maketrans("", "", string.punctuation))
        num_concepts = len(source.split())
        if num_concepts < min_concepts or num_concepts > max_concepts: 
            continue
        if extras == "cefr":
            source += " " + line["cefr"][0:2]
        if extras == "wsd":
            if line["wsd"]:
                source = line["wsd"]
            else:
                continue
        if extras == "srl":
            source = stitch_srl(source.split(), line["srl"])

        data.append({"input_ids": source,
                "labels": line["target"]})
    random.shuffle(data)
    return data


def predict(model: BartForConditionalGeneration, test_path:str, output_path:str, extras:str, n=1) -> None:
    ''' Write outputs for prediction. Controls are given by the input data '''
    model.to(DEVICE)
    if not output_path:
        output_path = DEFAULT_OUTPUT + test_path.split("/")[-1] + ".gen"

    data = load_data(test_path, extras)
    test_concepts = [d["input_ids"] for d in data]

    dct = TOKENIZER(test_concepts, max_length=64, padding="max_length", truncation=True, return_tensors="pt").to(DEVICE)
    hyp = model.generate(dct["input_ids"], max_length=65, num_beams=n, num_return_sequences=n)
    decoded_sents = TOKENIZER.batch_decode(hyp, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    with open(output_path, "w") as output_file:
        output_writer = csv.writer(output_file)
        outputs = []
        for e in range(len(decoded_sents)):
            outputs.append(decoded_sents[e].strip())
            if len(outputs) == n:
                output_writer.writerow(outputs)
                outputs = []
    print (str(len(decoded_sents)) + " outputs written to " + output_path)


def data_collator(features:list):
    ''' Data collator based on HF '''
    inputs = [f["input_ids"] for f in features]

    # very strange way of doing this, but it's what HF does
    batch = TOKENIZER(inputs, max_length=64, padding='max_length', truncation=True)
    with TOKENIZER.as_target_tokenizer():
        labels = TOKENIZER([f["labels"] for f in features], max_length=64, padding="max_length", truncation=True)
    batch["labels"] = labels["input_ids"]
    for k in batch:
        batch[k] = torch.tensor(batch[k]).to(DEVICE)
    
    return batch


def train_bart(data_location:str, model_path:str, output_dir:str, epochs:int, batch_size:int, n:int, config:str=None, extras:str=None, min_concepts:int=0, max_concepts:int=10) -> BartForConditionalGeneration:
    ''' Use HF trainer to do model training '''
    if model_path:
        model = BartForConditionalGeneration.from_pretrained(model_path)
    else:
        configuration = BartConfig() if not config else BartConfig(config)
        if type(configuration.vocab_size) == str:
            configuration.vocab_size = 50264
        model = BartForConditionalGeneration(configuration)

    model.to(DEVICE)

    train_dataset = load_data(data_location, extras, min_concepts, max_concepts)
    if n:
        train_dataset = train_dataset[:n]

    # defining training related arguments
    args = Seq2SeqTrainingArguments(output_dir=output_dir,
                        do_train=True,
                        #evaluation_strategy="epoch",    # not currently evaluating
                        per_device_train_batch_size=batch_size,
                        per_device_eval_batch_size=batch_size,
                        learning_rate=5e-5,
                        num_train_epochs=epochs,
                        logging_dir="./logs",
                        save_total_limit=5,
                        dataloader_pin_memory=False)


    # defining trainer using 🤗
    trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset, 
                eval_dataset=None)
 

    trainer.train()
    trainer.save_model(output_dir)
    return trainer.model


def main() -> None:
    ''' Can train, test, or both '''
    args = parse_args()
    if args.mode in ["train", "both"]:
        if not args.data_dir:
            print ("Need directory of training files to train...")
            return 1
        model = train_bart(data_location=args.data_dir, model_path=args.model_path, output_dir=args.output_dir, epochs=args.epochs, batch_size=args.batch_size, n=args.n, config=args.config, extras=args.extras, min_concepts=args.min_concepts, max_concepts=args.max_concepts)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.model_path)
    
    if args.mode in ["test", "both"]:
        predict(model, args.test_path, args.output_path, n=args.num_sentences, extras=args.extras)


if __name__ == "__main__":
    sys.exit(main())

