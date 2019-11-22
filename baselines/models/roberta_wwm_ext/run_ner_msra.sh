CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
export CUDA_VISIBLE_DEVICES="0"
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/chinese_roberta_wwm_ext_L-12_H-768_A-12
export GLUE_DIR=$CURRENT_DIR/../../glue/chineseGLUEdatasets/
TASK_NAME="msraner"

python run_ner.py \
  --task_name=$TASK_NAME \
  --do_train=true \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$GLUE_DIR/$TASK_NAME \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=5.0 \
  --output_dir=$CURRENT_DIR/${TASK_NAME}_output/
