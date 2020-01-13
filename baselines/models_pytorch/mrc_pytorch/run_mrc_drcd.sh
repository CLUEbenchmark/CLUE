#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=$OUTPUT_DIR/prev_trained_model/$MODEL_NAME
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="DRCD"

python run_mrc.py \
  --gpu_ids="0,1" \
  --train_epochs=2 \
  --n_batch=32 \
  --lr=3e-5 \
  --warmup_rate=0.1 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$BERT_DIR/pytorch_model.pth \
  --train_dir=$GLUE_DIR/$TASK_NAME/train_features.json \
  --train_file=$GLUE_DIR/$TASK_NAME/DRCD_training.json \
  --dev_dir1=$GLUE_DIR/$TASK_NAME/dev_examples.json \
  --dev_dir2=$GLUE_DIR/$TASK_NAME/dev_features.json \
  --dev_file=$GLUE_DIR/$TASK_NAME/DRCD_dev.json \
  --checkpoint_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/

python test_mrc.py \
  --gpu_ids="0" \
  --n_batch=32 \
  --max_seq_length=512 \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_restore_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
  --test_dir1=$GLUE_DIR/$TASK_NAME/test_examples.json \
  --test_dir2=$GLUE_DIR/$TASK_NAME/test_features.json \
  --test_file=$GLUE_DIR/$TASK_NAME/DRCD_test.json \

