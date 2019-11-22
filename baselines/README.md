<!--1. 数据集整体下载，解压到glue文件夹里-->
<!--  ```cd glue ```-->
<!--  ```wget https://storage.googleapis.com/chineseglue/chineseGLUEdatasets.v0.0.1.zip```-->

<!--    注：lcqmc数据集，请从<a href="http://icrc.hitsz.edu.cn/info/1037/1146.htm">这里</a>申请或搜索网络-->

<!--2. 训练模型-->
<!--  将预训练模型下载解压到对应的模型中prev_trained_model文件夹里-->
<!--  a. albert_xlarge-->
<!--  https://github.com/brightmart/albert_zh-->
<!--  b. bert-->
<!--  https://github.com/google-research/bert  -->
<!--  c. bert-wwm-ext  -->
<!--  https://github.com/ymcui/Chinese-BERT-wwm  -->
<!--  d. ernie  -->
<!--  https://github.com/ArthurRizar/tensorflow_ernie  -->
<!--  e. roberta    -->
<!--  https://github.com/brightmart/roberta_zh  -->
<!--  f. xlnet  -->
<!--  https://github.com/ymcui/Chinese-PreTrained-XLNet  -->
<!--  修改run_classifier.sh指定模型路径  -->
<!--  运行各个模型文件夹下不同任务对应的run_classifier_*.sh即可。如跑xnl运行： -->
<!--  ```sh run_classifier_xnli.sh```-->

1. 一键运行

    我们为您提供了可以“一键运行”的脚本来辅助您更快的在指定模型上运行特定任务。  
    
    以在 Bert 模型上运行“BQ 智能客服问句匹配”任务为例，您可以直接在 chineseGLUE/baselines/models/**bert**/ 下运行 run_classifier_**bq**.sh 脚本。

    ```bash
    cd chineseGLUE/baselines/models/bert/
    sh run_classifier_bq.sh
    ```
    
    该脚本将会自动下载“BQ 智能客服问句匹配”数据集（保存在chineseGLUE/baselines/glue/chineseGLUEdatasets/**bq**/ 文件夹下）和Bert模型（保存在 chineseGLUE/baselines/models/bert/prev_trained_model/ 下）。
    
    如果您想在其他模型上执行其他的任务，只需要在对应模型的目录下找到对应任务的执行脚本（ run_classifier_**??**.sh ），即可直接运行。
    
2. 测试效果

    1. TNEWS 文本分类 (Accuracy)
    
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge |   88.30 | 88.30   |batch_size=32, length=128, epoch=3 |
    | BERT-base | 89.80 | 89.78 | batch_size=32, length=128, epoch=3 |
    | BERT-wwm-ext-base | 89.88 | 89.81 |   batch_size=32, length=128, epoch=3 |
    | ERNIE-base    | 89.77 |89.83 | batch_size=32, length=128, epoch=3 |
    | RoBERTa-large |***90.00*** | ***89.91*** |    batch_size=16, length=128, epoch=3 |
    | XLNet-mid |86.14 | 86.26 |    batch_size=32, length=128, epoch=3 | 
    | RoBERTa-wwm-ext | 89.82 | 89.79 | batch_size=32, length=128, epoch=3 | 
    | RoBERTa-wwm-large-ext | ***90.05*** | ***90.11*** |   batch_size=16, length=128, epoch=3 | 

    2. XNLI 自然语言推理 (Accuracy)
    
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge |74.0？ |74.0？   |batch_size=64, length=128, epoch=2 |
    | ALBERT-base | 77.0 |  77.1 |batch_size=64, length=128, epoch=2 |
    | ALBERT-large | 78.0 | 77.5    |batch_size=64, length=128, epoch=2 |
    | BERT-base | 77.80 | 77.8  | batch_size=64, length=128, epoch=2 |
    | BERT-wwm-ext-base | 79.4 | 78.7 | batch_size=64, length=128, epoch=2 |
    | ERNIE-base    | 79.7  |78.6 | batch_size=64, length=128, epoch=2 |
    | RoBERTa-large |***80.2*** |***79.9*** |   batch_size=64, length=128, epoch=2 |
    | XLNet-mid |79.2 | 78.7 |  batch_size=64, length=128, epoch=2 | 
    | RoBERTa-wwm-ext   | 79.56 | 79.28 |   batch_size=64, length=128, epoch=2 | 
    | RoBERTa-wwm-large-ext | ***80.20*** | ***80.04*** |   batch_size=16, length=128, epoch=2 | 
    
    3. LCQMC  语义相似度匹配 (Accuracy)
    
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge | 89.00 | 86.76 |batch_size=64, length=128, epoch=3 |
    | BERT-base | 89.4  | 86.9  | batch_size=64, length=128, epoch=3 |
    | BERT-wwm-ext-base |89.1   | ***87.3*** |  batch_size=64, length=128, epoch=3 |
    | ERNIE-base    | 89.8  | 87.2 | batch_size=64, length=128, epoch=3|
    | RoBERTa-large |***89.9***  | 87.2|    batch_size=64, length=128, epoch=3 |
    | XLNet-mid | 86.14 | 85.98 |   batch_size=32, length=128, epoch=3 | 
    | RoBERTa-wwm-ext   | 89.08 | 86.33 |   batch_size=64, length=128, epoch=3 | 
    | RoBERTa-wwm-large-ext | 89.79 | 86.82 |   batch_size=16, length=128, epoch=3 | 

    4. INEWS 互联网情感分析 (Accuracy)
    
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge |83.70   | 81.90    |batch_size=32, length=512, epoch=8 |
    | BERT-base | 81.29 | 82.70 | batch_size=16, length=512, epoch=3 |
    | BERT-wwm-ext-base | 81.93 | 83.46 |   batch_size=16, length=512, epoch=3 |
    | ERNIE-base    | ***84.50***   |***85.14*** | batch_size=16, length=512, epoch=3 |
    | RoBERTa-large |81.90 | 84.00 |    batch_size=4, length=512, epoch=3 |
    | XLNet-mid |82.00 | 84.00 |    batch_size=8, length=512, epoch=3 | 
    | RoBERTa-wwm-ext   | 82.98 | 82.28 |   batch_size=16, length=512, epoch=3 | 
    | RoBERTa-wwm-large-ext |83.73 | 82.78 |    batch_size=4, length=512, epoch=3 |

    5. DRCD 繁体阅读理解 (F1, EM)
    
    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base |F1:92.30 EM:86.60 | F1:91.46 EM:85.49 |    batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | BERT-wwm-ext-base |F1:93.27 EM:88.00 | F1:92.63 EM:87.15 |    batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ERNIE-base    |F1:92.78 EM:86.85 | F1:92.01 EM:86.03 |    batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ALBERT-large  |F1:93.90 EM:88.88 | F1:93.06 EM:87.52 |    batch=32, length=512, epoch=3 lr=2e-5 warmup=0.05 |
    | ALBERT-xlarge |F1:94.63 EM:89.68 | F1:94.70 EM:89.78 |    batch_size=32, length=512, epoch=3 lr=2.5e-5 warmup=0.06 |
    | ALBERT-tiny   |F1:81.51 EM:71.61 | F1:80.67 EM:70.08 |    batch=32, length=512, epoch=3 lr=2e-4 warmup=0.1 |
    | RoBERTa-large |F1:94.93 EM:90.11 | F1:94.25 EM:89.35 |    batch=32, length=256, epoch=2 lr=3e-5 warmup=0.1|
    | xlnet-mid |F1:92.08 EM:84.40 | F1:91.44 EM:83.28 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |  
    | RoBERTa-wwm-ext   |F1:94.26 EM:89.29 | F1:93.53 EM:88.12 |    batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1|
    | RoBERTa-wwm-large-ext |***F1:95.32 EM:90.54*** | ***F1:95.06 EM:90.70*** | batch=32, length=512, epoch=2 lr=2.5e-5 warmup=0.1 | 
    
    6. CMRC2018 阅读理解 (F1, EM)
    
    | 模型 | 开发集（dev) | 测试集（test) |  训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base	|F1:85.48 EM:64.77 | F1:87.17 EM:69.72 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | BERT-wwm-ext-base	|F1:86.68 EM:66.96 |F1:88.78 EM:73.23|	batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ERNIE-base	|F1:87.30 EM:66.89 | F1:89.62 EM:73.32 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
    | ALBERT-large	| F1:87.86 EM:67.75 |F1:90.17 EM:73.66| epoch3, batch=32, length=512, lr=2e-5, warmup=0.05 |
    | ALBERT-xlarge	| F1:88.66 EM:68.90 |F1:90.92 EM:75.22| epoch3, batch=32, length=512, lr=2e-5, warmup=0.1 |
    | ALBERT-tiny	| F1:73.95 EM:48.31 |F1:75.73 EM:53.68| epoch3, batch=32, length=512, lr=2e-4, warmup=0.1 |
    | RoBERTa-large	| F1:88.61 EM:69.94 |F1:90.94 EM:76.11| epoch2, batch=32, length=256, lr=3e-5, warmup=0.1 |
    | xlnet-mid	|F1:85.63 EM:65.31 | F1:86.09 EM:66.51 | epoch2, batch=32, length=512, lr=3e-5, warmup=0.1 |
    | RoBERTa-wwm-ext	|F1:87.28 EM:67.89 | F1:89.74 EM:73.89 | epoch2, batch=32, length=512, lr=3e-5, warmup=0.1 |
    | RoBERTa-wwm-large-ext	|***F1:89.42 EM:70.59*** | ***F1:91.56 EM:76.58*** | epoch2, batch=32, length=512, lr=2.5e-5, warmup=0.1 |

    7. BQ 智能客服问句匹配 (Accuracy)

    | 模型 | 开发集（dev） | 测试集（test） | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | BERT-base | 85.86 | 85.08 | batch_size=64, length=128, epoch=3 |
    | BERT-wwm-ext-base | 86.05 | ***85.21*** |batch_size=64, length=128, epoch=3 |
    | ERNIE-base | 85.92 | 84.47 | batch_size=64, length=128, epoch=3 |
    | RoBERTa-large | 85.68 | 85.20 | batch_size=8, length=128, epoch=3 |
    | XLNet-mid | 79.81 | 77.85 | batch_size=32, length=128, epoch=3 |
    | ALBERT-xlarge |   85.21 | 84.21 | batch_size=16, length=128, epoch=3 |
    | ALBERT-tiny | 82.04 | 80.76 | batch_size=64, length=128, epoch=5 |
    | RoBERTa-wwm-ext | 85.31 | 84.02 | batch_size=64, length=128, epoch=3 |
    | RoBERTa-wwm-large-ext | ***86.34*** | 84.90 | batch_size=16, length=128, epoch=3 |

    8. MSRANER 命名实体识别 (F1)

    | 模型 | 测试集（test） | 训练参数 |
    | :----: | :----: | :----: |
    | BERT-base | 95.38 | batch_size=16, length=256, epoch=5, lr=2e-5 |
    | BERT-wwm-ext-base | 95.26 | batch_size=16, length=256, epoch=5, lr=2e-5 |
    | ERNIE-base | 95.17 | batch_size=16, length=256, epoch=5, lr=2e-5 |
    | RoBERTa-large | ***96.07***   | batch_size=8, length=256, epoch=5, lr=2e-5 |
    | XLNet-mid | - | - |
    | ALBERT-xlarge | - | - |
    | ALBERT-tiny | - | - |
    | RoBERTa-wwm-ext | 95.06   | batch_size=16, length=256, epoch=5, lr=2e-5 |
    | RoBERTa-wwm-large-ext | 95.32 | batch_size=8, length=256, epoch=5, lr=2e-5 |

    9. THUCNEWS 长文本分类 (Accuracy)

    | 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
    | :----:| :----: | :----: | :----: |
    | ALBERT-xlarge | 95.74 | 95.45 |batch_size=32, length=512, epoch=8 |
    | ALBERT-tiny | 92.63 | 93.54 | batch_size=32, length=512, epoch=8 |
    | BERT-base     | 95.28 | 95.35 | batch_size=8, length=128, epoch=3 |
    | BERT-wwm-ext-base | 95.38 | 95.57 |   batch_size=8, length=128, epoch=3 |
    | ERNIE-base    | 94.35 | 94.90 | batch_size=16, length=256, epoch=3 |
    | RoBERTa-large | 94.52 | 94.56 |       batch_size=2, length=256, epoch=3 |
    | XLNet-mid     | - | 94.52 |   batch_size=16, length=128, epoch=3 |
    | RoBERTa-wwm-ext       | 95.59 | 95.52 |       batch_size=16, length=256, epoch=3 |
    | RoBERTa-wwm-large-ext | ***96.10*** | ***95.93*** |    batch_size=32, length=512, epoch=8 |
    
