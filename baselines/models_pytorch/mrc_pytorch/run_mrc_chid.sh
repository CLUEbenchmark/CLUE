#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=$OUTPUT_DIR/prev_trained_model/$MODEL_NAME
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="CHID"

python run_multichoice_mrc.py \
  --gpu_ids="0,1,2,3" \
  --num_train_epochs=3 \
  --train_batch_size=24 \
  --predict_batch_size=24 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.06 \
  --max_seq_length=64 \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$BERT_DIR/pytorch_model.pth \
  --input_dir=$GLUE_DIR/$TASK_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --train_file=$GLUE_DIR/$TASK_NAME/train.json \
  --train_ans_file=$GLUE_DIR/$TASK_NAME/train_answer.json \
  --predict_file=$GLUE_DIR/$TASK_NAME/dev.json \
  --predict_ans_file=$GLUE_DIR/$TASK_NAME/dev_answer.json

python test_multichoice_mrc.py \
  --gpu_ids="0" \
  --predict_batch_size=24 \
  --max_seq_length=64 \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --input_dir=$GLUE_DIR/$TASK_NAME/ \
  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --predict_file=$GLUE_DIR/$TASK_NAME/test.json \
