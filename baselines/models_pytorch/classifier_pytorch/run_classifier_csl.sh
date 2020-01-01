#!/usr/bin/env bash
# @Author: bo.shi
# @Date:   2019-11-04 09:56:36
# @Last Modified by:   bo.shi
# @Last Modified time: 2020-01-01 11:41:49

TASK_NAME="csl"
MODEL_NAME="bert-base-chinese"
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_PRETRAINED_MODELS_DIR=$CURRENT_DIR/prev_trained_model
export BERT_WWM_DIR=$BERT_PRETRAINED_MODELS_DIR/$MODEL_NAME
export GLUE_DATA_DIR=$CURRENT_DIR/CLUEdatasets

# download and unzip dataset
if [ ! -d $GLUE_DATA_DIR ]; then
  mkdir -p $GLUE_DATA_DIR
  echo "makedir $GLUE_DATA_DIR"
fi
cd $GLUE_DATA_DIR
if [ ! -d $TASK_NAME ]; then
  mkdir $TASK_NAME
  echo "makedir $GLUE_DATA_DIR/$TASK_NAME"
fi
cd $TASK_NAME
if [ ! -f "train.json" ] || [ ! -f "dev.json" ] || [ ! -f "test.json" ]; then
  rm *
  wget https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip
  unzip csl_public.zip
  rm csl_public.zip
else
  echo "data exists"
fi
echo "Finish download dataset."

# make output dir
if [ ! -d $CURRENT_DIR/${TASK_NAME}_output ]; then
  mkdir -p $CURRENT_DIR/${TASK_NAME}_output
  echo "makedir $CURRENT_DIR/${TASK_NAME}_output"
fi

# run task
cd $CURRENT_DIR
echo "Start running..."
if [ $# == 0 ]; then
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_train \
      --do_eval \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=256 \
      --per_gpu_train_batch_size=8 \
      --per_gpu_eval_batch_size=8 \
      --learning_rate=1e-5 \
      --num_train_epochs=5.0 \
      --logging_steps=2500 \
      --save_steps=2500 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
elif [ $1 == "predict" ]; then
    echo "Start predict..."
    python run_classifier.py \
      --model_type=bert \
      --model_name_or_path=$MODEL_NAME \
      --task_name=$TASK_NAME \
      --do_predict \
      --do_lower_case \
      --data_dir=$GLUE_DATA_DIR/${TASK_NAME}/ \
      --max_seq_length=256 \
      --per_gpu_train_batch_size=8 \
      --per_gpu_eval_batch_size=8 \
      --learning_rate=1e-5 \
      --num_train_epochs=5.0 \
      --logging_steps=2500 \
      --save_steps=2500 \
      --output_dir=$CURRENT_DIR/${TASK_NAME}_output/ \
      --overwrite_output_dir \
      --seed=42
fi

