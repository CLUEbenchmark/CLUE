CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")
TASK_NAME="tnews"
export PREV_TRAINED_MODEL_DIR=gs://models_zxw/prev_trained_models/nlp/albert-tiny/albert_tiny_489k
export DATA_DIR=gs://data_zxw/nlp/chineseGLUEdatasets.v0.0.1/hard_${TASK_NAME}_1
export OUTPUT_DIR=gs://models_zxw/fine_tuning_models/nlp/albert-tiny/tpu/${TASK_NAME}/$CURRENT_TIME

python $CURRENT_DIR/../run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$PREV_TRAINED_MODEL_DIR/vocab.txt \
  --bert_config_file=$PREV_TRAINED_MODEL_DIR/albert_config_tiny.json \
  --init_checkpoint=$PREV_TRAINED_MODEL_DIR/albert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=6e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR \
  --num_tpu_cores=8 --use_tpu=True --tpu_name=grpc://172.20.0.2:8470
