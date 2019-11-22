CURRENT_DIR=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
CURRENT_TIME=$(date "+%Y%m%d-%H%M%S")
TASK_NAME="tnews"
export XLNET_DIR=gs://models_zxw/prev_trained_models/nlp/xlnet-base/chinese_xlnet_base_L-12_H-768_A-12
export DATA_DIR=gs://data_zxw/nlp/chineseGLUEdatasets.v0.0.1/hard_${TASK_NAME}_1
export OUTPUT_DIR=gs://models_zxw/fine_tuning_models/nlp/xlnet-base/chinese_xlnet_base_L-12_H-768_A-12/tpu/$TASK_NAME/$CURRENT_TIME

python $CURRENT_DIR/../run_classifier.py \
    --spiece_model_file=${CURRENT_DIR}/../spiece.model \
    --model_config_path=${XLNET_DIR}/xlnet_config.json \
    --init_checkpoint=${XLNET_DIR}/xlnet_model.ckpt \
    --task_name=$TASK_NAME \
    --do_train=True \
    --do_eval=True \
    --eval_all_ckpt=True \
    --uncased=False \
    --data_dir=$DATA_DIR \
    --output_dir=${OUTPUT_DIR} \
    --model_dir=${OUTPUT_DIR} \
    --train_batch_size=16 \
    --eval_batch_size=8 \
    --num_hosts=1 \
    --num_core_per_host=8 \
    --num_train_epochs=3 \
    --max_seq_length=128 \
    --learning_rate=1e-5 \
    --save_steps=1000 \
    --use_tpu=True --tpu=grpc://192.168.0.2:8470
