# Controlled Language Generation for Language Learning Items

This is the repository for the paper <a href="https://arxiv.org/abs/2211.15731">Controlled Language Generation for Language Learning Items</a> at the Industry Track, EMNLP 2022. The code is based heavily on HuggingFace's sequence-to-sequence Trainer examples.

## Requirements
Scripts were tested with python 3.9 and the transformers package. Nothing else should be required.

## Data
The data is provided as jsonlines objects containing relevant fields for concept-to-sequence generation with control. The files require <a href="https://git-lfs.com/">Git LFS</a>.

## Training

To train, call the concept2seq.py script with --mode train, along with the required parameters.

```
# Set a root directory
r=/home/nlp-text/dynamic/kstowe/github/concept-control-gen/
data_json=${r}/data/concept2seq_train.jsonl

# Substitute in your python
/home/conda/kstowe/envs/pretrain/bin/python $r/concept2seq.py \
    --mode train \
    --data_dir $data_json \
    --output_dir $r/models/c2s_test \
    --epochs 3 \
    --batch_size 32 \
    --model_path facebook/bart-base \
#    --extras srl \
#    --n 100
```

## Prediction

Prediction works similarly, using the supported parameters.

```
# Set a nice root
r=/home/nlp-text/dynamic/kstowe/github/concept-control-gen/

/home/conda/kstowe/envs/pretrain/bin/python $r/concept2seq.py \
        --mode test \
        --output_path $r/outputs/test.txt \
        --test_path ${r}/data/concept2seq_test.jsonl \
        --model_path kevincstowe/concept2seq
```
