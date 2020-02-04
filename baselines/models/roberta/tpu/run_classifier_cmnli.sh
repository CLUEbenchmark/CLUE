#!/usr/bin/env bash
CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")
CLUE_DATA_DIR=gs://data_zxw/nlp/CLUE
CLUE_PREV_TRAINED_MODEL_DIR=gs://models_zxw/prev_trained_models/nlp
CLUE_OUTPUT_DIR=gs://models_zxw/fine_tuning_models/nlp
run_task() {
TASK_NAME=$1
MODEL_NAME=$2
DATA_DIR=$CLUE_DATA_DIR/${TASK_NAME}_public
PREV_TRAINED_MODEL_DIR=$CLUE_PREV_TRAINED_MODEL_DIR/$MODEL_NAME
MAX_SEQ_LENGTH=$3
TRAIN_BATCH_SIZE=$4
LEARNING_RATE=$5
NUM_TRAIN_EPOCHS=$6
OUTPUT_DIR=$CLUE_OUTPUT_DIR/$MODEL_NAME/$CURRENT_TIME
COMMON_ARGS="
      --task_name=$TASK_NAME \
      --data_dir=$DATA_DIR \
      --vocab_file=$PREV_TRAINED_MODEL_DIR/vocab.txt \
      --bert_config_file=$PREV_TRAINED_MODEL_DIR/roberta_config_tiny.json \
      --init_checkpoint=$PREV_TRAINED_MODEL_DIR/roberta_model.ckpt \
      --max_seq_length=$MAX_SEQ_LENGTH \
      --train_batch_size=$TRAIN_BATCH_SIZE \
      --learning_rate=$LEARNING_RATE \
      --num_train_epochs=$NUM_TRAIN_EPOCHS \
      --output_dir=$OUTPUT_DIR
"
echo "Start running..."
python $CURRENT_DIR/../run_classifier.py \
      $COMMON_ARGS \
      --do_train=true \
      --do_eval=false \
      --do_predict=false 

echo "Start predict..."
python $CURRENT_DIR/../run_classifier.py \
      $COMMON_ARGS \
      --do_train=false \
      --do_eval=true \
      --do_predict=true 
}

run_task cmnli roberta_tiny_normal 128 16 1e-5 3

