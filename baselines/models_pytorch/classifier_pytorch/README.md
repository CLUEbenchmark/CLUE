# CLUE_pytorch

中文语言理解测评基准(Language Understanding Evaluation benchmark for Chinese)

**备注**：此版本为个人开发版(目前支持所有的分类型任务)，正式版见https://github.com/CLUEbenchmark/CLUE

## 代码目录说明

```text
├── CLUEdatasets   #　存放数据
|  └── tnews　　　
|  └── wsc　
|  └── ...
├── metrics　　　　　　　　　# metric计算
|  └── clue_compute_metrics.py　　　
├── outputs              # 模型输出保存
|  └── tnews_output
|  └── wsc_output　
|  └── ...
├── prev_trained_model　# 预训练模型
|  └── albert_base
|  └── bert-wwm
|  └── ...
├── processors　　　　　# 数据处理
|  └── clue.py
|  └── ...
├── tools　　　　　　　　#　通用脚本
|  └── progressbar.py
|  └── ...
├── transformers　　　# 主模型
|  └── modeling_albert.py
|  └── modeling_bert.py
|  └── ...
├── convert_albert_original_tf_checkpoint_to_pytorch.py　#　模型文件转换
├── run_classifier.py       # 主程序
├── run_classifier_tnews.sh   #　任务运行脚本
├── download_clue_data.py   # 数据集下载
```
### 依赖模块

- pytorch=1.1.0
- boto3=1.9
- regex
- sacremoses
- sentencepiece
- python3.7+

### 运行方式

**1. 下载CLUE数据集，运行以下命令：**
```python
python download_clue_data.py --data_dir=./CLUEdatasets --tasks=all
```
上述命令默认下载全CLUE数据集，你也可以指定`--tasks`进行下载对应任务数据集，默认存在在`./CLUEdatasets/{对应task}`目录下。

**2. 若下载对应tf模型权重(若下载为pytorch权重，则跳过该步)，运行转换脚本，比如转换`albert_base_tf`:**

```python
python convert_albert_original_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path=./prev_trained_model/albert_base_tf \
      --bert_config_file=./prev_trained_model/albert_base_tf/albert_config_base.json \
      --pytorch_dump_path=./prev_trained_model/albert_base/pytorch_model.bin
```
**注意**: 当转换完模型(包括下载的pytorch模型权重)之后，需要在对应的文件夹内存放`config.json`和`vocab.txt`文件，比如：

```text
├── prev_trained_model　# 预训练模型
|  └── bert-base
|  | └── vocab.txt
|  | └── config.json
|  | └── pytorch_model.bin

```
**3. 直接运行对应任务sh脚本，如：**

```shell
sh run_classifier_tnews.sh
```
**4. 评估**

当前默认使用最后一个checkpoint模型作为评估模型，你也可以指定`--predict_checkpoints`参数进行对应的checkpoint进行评估，比如：
```python
CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/prev_trained_model/bert-base
export GLUE_DIR=$CURRENT_DIR/CLUEdatasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="copa"

python run_classifier.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_predict \
  --predict_checkpoints=100 \
  --do_lower_case \
  --data_dir=$GLUE_DIR/${TASK_NAME}/ \
  --max_seq_length=128 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=1e-5 \
  --num_train_epochs=2.0 \
  --logging_steps=50 \
  --save_steps=50 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
```

### 模型列表

```
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # xlnet_base xlnet_mid xlnet_large
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # roberta_base roberta_wwm roberta_wwm_ext roberta_wwm_large_ext
    'roberta': (BertConfig, BertForSequenceClassification, BertTokenizer),
    # albert_tiny albert_base albert_large albert_xlarge
    'albert': (BertConfig, AlbertForSequenceClassification, BertTokenizer)
}
```
**注意**: bert ernie bert_wwm bert_wwwm_ext等模型只是权重不一样，而模型本身主体一样，因此参数`model_type=bert`其余同理。

### 结果

当前按照https://github.com/CLUEbenchmark/CLUE  提供的参数，除了**COPA**任务无法复现，其余任务基本保持一致。




