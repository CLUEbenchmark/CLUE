CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")
TASK_NAME="inews"
export PREV_TRAINED_MODEL_DIR=gs://models_zxw/prev_trained_models/nlp/al/bert-base/chinese_L-12_H-768_A-12/
export DATA_DIR=gs://data_zxw/nlp/chineseGLUEdatasets.v0.0.1/$TASK_NAME
export OUTPUT_DIR=gs://models_zxw/fine_tuning_models/nlp/bert-base/chinese_L-12_H-768_A-12/tpu/$TASK_NAME/$CURRENT_TIME

python $CURRENT_DIR/../run_classifier.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$PREV_TRAINED_MODEL_DIR/vocab.txt \
  --bert_config_file=$PREV_TRAINED_MODEL_DIR/bert_config.json \
  --init_checkpoint=$PREV_TRAINED_MODEL_DIR/bert_model.ckpt \
  --max_seq_length=512 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=8.0 \
  --output_dir=$OUTPUT_DIR \
  --num_tpu_cores=8 --use_tpu=True --tpu_name=grpc://10.1.101.2:8470
