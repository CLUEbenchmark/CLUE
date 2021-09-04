#!/usr/bin/env bash

CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export MODEL_NAME=roberta_wwm_ext_base
export OUTPUT_DIR=$CURRENT_DIR/check_points
export BERT_DIR=$OUTPUT_DIR/prev_trained_model/$MODEL_NAME
export GLUE_DIR=$CURRENT_DIR/mrc_data # set your data dir
TASK_NAME="c3"

# download and unzip dataset
if [ ! -d $GLUE_DIR ]; then
  mkdir -p $GLUE_DIR
  echo "makedir $GLUE_DIR"
fi
cd $GLUE_DIR
if [ ! -d $TASK_NAME ]; then
  mkdir $TASK_NAME
  echo "makedir $GLUE_DIR/$TASK_NAME"
fi
cd $TASK_NAME
if [ ! -f "d-train.json" ] || [ ! -f "m-train.json" ] || [ ! -f "d-dev.json" ] || [ ! -f "test1.1.json" ]; then
  rm *
  wget https://storage.googleapis.com/cluebenchmark/tasks/c3_public.zip
  unzip c3_public.zip
  rm c3_public.zip
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
python run_c3.py \
  --gpu_ids="0,1,2,3" \
  --num_train_epochs=8 \
  --train_batch_size=24 \
  --eval_batch_size=24 \
  --gradient_accumulation_steps=4 \
  --learning_rate=2e-5 \
  --warmup_proportion=0.05 \
  --max_seq_length=512 \
  --do_train \
  --do_eval \
  --task_name=$TASK_NAME \
  --vocab_file=$BERT_DIR/vocab.txt \
  --bert_config_file=$BERT_DIR/bert_config.json \
  --init_checkpoint=$BERT_DIR/pytorch_model.pth \
  --data_dir=$GLUE_DIR/$TASK_NAME/ \
  --output_dir=$OUTPUT_DIR/$TASK_NAME/$MODEL_NAME/ \
