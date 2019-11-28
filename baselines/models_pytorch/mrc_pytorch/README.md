# CLUE MRC pytorch

**详细信息见于https://github.com/chineseGLUE/chineseGLUE**

## 代码目录说明

```text
├── mrc_data   #　存放数据
|  └── CHID　　　
|  └── DRCD　
|  └── CMRC2018
|  └── ...
├── preprocess          # 预处理和评测
├── check_points        # 预训练模型和模型输出保存
|  └── prev_trained_model
|  └── CHID　　　
|  └── DRCD　
|  └── CMRC2018
|  └── ...
├── tools　　　　　　　　#　通用脚本
├── convert_tf_checkpoint_to_pytorch.py　#　模型文件转换
├── pytorch_modeling.py　#　模型文件
├── run_mrc.py       # 主程序
├── run_mrc_xxxx.sh   #　任务运行脚本
```
### 依赖模块

- pytorch>=1.1.0

### 运行

1. 若下载对应tf模型权重，则运行转换脚本:
```
python convert_tf_checkpoint_to_pytorch.py \
      --tf_checkpoint_path=.check_points/prev_trained_model/roberta_wwm_ext_large/bert_model.ckpt \
      --bert_config_file=.check_points/prev_trained_model/roberta_wwm_ext_large/bert_config.json \
      --pytorch_dump_path=.check_points/prev_trained_model/roberta_wwm_ext_large/pytorch_model.pth
```
**注意**: 当转换完模型之后，需要在对应的文件夹内存放`bert_config.json`和`vocab.txt`文件

2. 直接运行对应任务sh脚本，如：

```shell
sh run_mrc_cmrc2018.sh
```
**注意**: 请根据需求调整参数和路径





