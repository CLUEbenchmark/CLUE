# ChineseGLUE
Language Understanding Evaluation benchmark for Chinese: datasets, baselines, pre-trained models, corpus and leaderboard

中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜。  

我们会选择一系列有一定代表性的任务对应的数据集，做为我们测试基准的数据集。这些数据集会覆盖不同的任务、数据量、任务难度。

中文任务基准测评(ChineseGLUE)-排行榜 Leaderboard
---------------------------------------------------------------------
#####  排行榜会定期更新                     数据来源: https://github.com/chineseGLUE/chineseGLUE


#### 分类任务(v1版本,正式版)

| 模型   | Score  | 参数    | LCQMC'  | TNEWS'  | IFYTEK'   | MNLI-m  | MNLI-mm  | XNLI     | COPA | WSC | -  |
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>        | - | 108M | 74.89% | 55.58%  | 60.29% | 79.39%  | 79.76% | 77.8%  | - | -  | -      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a>      | - | 108M  | 76.72% | 56.84%  | -  | - | -| 78.7%  | -   | -  | -      |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>         | - | 108M  | 77.94% | 58.23% | - | - | - | 78.6%  | -  | -  | -      |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a>      | - | 334M  | 77.57% | 57.05%  | -  | -  |  - | 79.9%   | -  | -   | -       |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>  | - | 200M | - | - | -  | -  | -   | 78.7% | -      | -   | -     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a>      | - | 59M   | -  | - | - | - | - | - | -  | -  | -     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a>        | - | 4M |76.57% | 53.35% | -  | -   | -  | - | -  | -   | -     |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>   | - | 108M  | 77.43% | 56.86% | - | - | - | 79.3%  | -  | - | -      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | - | 330M | **78.02%** | **58.61%** | 62.98% | 83.4% | 83.42% | **80.0%** | - | - | - |


注：' 代表对原数据集筛选后获得，数据集与原数据集不同；TNEWS:文本分类(Acc)；LCQMC:语义相似度(Acc)；XNLI/MNLI:自然语言推理(Acc),MNLI-m:MNLI-matched，MNLI-mm:MNLI-mismatched；

DRCD & CMRC2018:抽取式阅读理解(F1, EM)；CHID:成语多分类阅读理解(Acc)；BQ:智能客服问句匹配(Acc)；MSRANER:命名实体识别(F1)；iFLYTEK:长文本分类(Acc)；

Score是通过计算1-9数据集得分平均值获得；


#### 阅读理解任务

| 模型 | Score | 参数 | DRCD | CMRC2018 | CHID |
| :----:| :----: | :----: | :----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>	| 79.08 | 108M | 85.49 	| 69.72 | 82.04 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a> | - | 108M | 87.15 | 73.23 | - |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>	| - | 108M | 86.03 | 73.32 | - |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a> | 83.32 | 334M 	| 89.35 | 76.11 | 84.5 |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>	| - | 209M | 83.28 | 66.51  | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xlarge</a> | - | 59M | 89.78 | 75.22 | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a> | - | - | - | - | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a> | - | 1.8M | 70.08 | 53.68 | - |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>  | 81.88 | 108M  | 88.12 | 73.89 | 83.62 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | ***84.22*** | 330M |	***90.70*** |	***76.58*** | ***85.37*** |

注：阅读理解上述指标中F1和EM共存的情况下，取EM为最终指标


#### 分类任务(vO版本，初版)

| 模型   | Score  | 参数    | TNEWS  | LCQMC  | XNLI   | INEWS  | BQ     | MSRANER | THUCNEWS | iFLYTEKData |
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>        | 84\.57 | 108M  | 89\.78 | 86\.9  | 77\.8  | 82\.7  | 85\.08 | 95\.38  | 95\.35   | 63\.57      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a>      | 84\.89 | 108M  | 89\.81 | ***87\.3***  | 78\.7  | 83\.46 | ***85\.21*** | 95\.26  | 95\.57   | 63\.83      |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>         | 84\.63 | 108M  | 89\.83 | 87\.2  | 78\.6  | ***85\.14*** | 84\.47 | 95\.17  | 94\.9    | 61\.75      |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a>      | 85\.08 | 334M  | 89\.91 | 87\.2  | 79\.9  | 84     | 85\.2  | ***96\.07***  | 94\.56   | 63\.8       |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>          | 81\.07 | 209M  | 86\.26 | 85\.98 | 78\.7  | 84     | 77\.85 | \-      | 94\.54   | 60\.16      |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xlarge</a>      | 84\.08 | 59M   | 88\.3  | 86\.76 | 74\.0? | 82\.4  | 84\.21 | 89\.51  | 95\.45   | 61\.94      |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a>        | 78\.22 | 1\.8M | 87\.1  | 85\.4  | 68     | 81\.4  | 80\.76 | 84\.77  | 93\.54   | 44\.83      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>   | 84\.55 | 108M  | 89\.79 | 86\.33 | 79\.28 | 82\.28 | 84\.02 | 95\.06  | 95\.52   | 64\.18      |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | ***85\.13*** | 330M  | ***90\.11*** | 86\.82 | ***80\.04*** | 82\.78 | 84\.9  | 95\.32  | ***95\.93***   | ***65\.19***      |


ChineseGLUE的定位 Vision
---------------------------------------------------------------------
为更好的服务中文语言理解、任务和产业界，做为通用语言模型测评的补充，通过完善中文语言理解基础设施的方式来促进中文语言模型的发展

*** 2019-10-13: 新增评测官网入口; INEWS基线模型 ***

  <a href="http://106.13.187.75:8003/"> 评测入口</a>

Why do we need a benchmark for Chinese lanague understand evaluation?

为什么我们需要一个中文任务的基准测试？ 
---------------------------------------------------------------------
首先，中文是一个大语种，有其自身的特定、大量的应用。

    如中文使用人数近14亿，是联合国官方语言之一，产业界有大量的的朋友在做中文的任务。
    中文是象形文字，有文字图形；字与字之间没有分隔符，不同的分词(分字或词)会影响下游任务。

其次，相对于英文的数据集，中文的公开可用的数据集还比较少。
     
     很多数据集是非公开的或缺失基准测评的；多数的论文描述的模型是在英文数据集上做的测试和评估，那么对于中文效果如何？不得而知。

再次，语言理解发展到当前阶段，预训练模型极大的促进了自然语言理解。

     不同的预训练模型相继产生，但不少最先进(state of the art)的模型，并没有官方的中文的版本，也没有对这些预训练模型在不同任务上的公开测试，
     导致技术的发展和应用还有不少距离，或者说技术应用上的滞后。

那么，如果有一个中文任务的基准测试，包含一批大众能广泛使用和测评的数据集、适用中文任务的特点、能紧跟当前世界技术的发展，
     
     能缓解当前中文任务的一些问题，并促进相关应用的发展。

中文任务的基准测试-内容体系 Contents
--------------------------------------------------------------------
Language Understanding Evaluation benchmark for Chinese(ChineseGLUE) got ideas from GLUE, which is a collection of 

resources for training, evaluating, and analyzing natural language understanding systems. ChineseGLUE consists of: 

##### 1）中文任务的基准测试，覆盖多个不同程度的语言任务 

A benchmark of several sentence or sentence pair language understanding tasks. 
Currently the datasets used in these tasks are come from public. We will include datasets with private test set before the end of 2019.

##### 2）公开的排行榜 

A public leaderboard for tracking performance. You will able to submit your prediction files on these tasks, each task will be evaluated and scored, a final score will also be available.

##### 3）基线模型，包含开始的代码、预训练模型 

baselines for ChineseGLUE tasks. baselines will be available in TensorFlow,PyTorch,Keras and PaddlePaddle.

##### 4）语料库，用于语言建模、预训练或生成型任务 

A huge amount of raw corpus for pre-train or language modeling research purpose. It will contains around 10G raw corpus in 2019; 

In the first half year of 2020, it will include at least 30G raw corpus; By the end of 2020, we will include enough raw corpus, such as 100G, so big enough that you will need no more raw corpus for general purpose language modeling.
You can use it for general purpose or domain adaption, or even for text generating. when you use for domain adaption, you will able to select corpus you are interested in.

数据集介绍与下载 Introduction of datasets 
--------------------------------------------------------------------
##### 1. LCQMC 口语化描述的语义相似度任务 Semantic Similarity Task
输入是两个句子，输出是0或1。其中0代表语义不相似，1代表语义相似。

        数据量：训练集(238,766)，验证集(8,802)，测试集(12,500)
        例子： 
         1.聊天室都有哪些好的 [分隔符] 聊天室哪个好 [分隔符] 1
         2.飞行员没钱买房怎么办？ [分隔符] 父母没钱买房子 [分隔符] 0

##### 2. XNLI 语言推断任务 Natural Language Inference
跨语言理解的数据集，给定一个前提和假设，判断这个假设与前提是否具有蕴涵、对立、中性关系。
                
        数据量：训练集(392,703)，验证集(2,491)，测试集(5,011)
        例子： 
         1.从 概念 上 看 , 奶油 收入 有 两 个 基本 方面 产品 和 地理 .[分隔符] 产品 和 地理 是 什么 使 奶油 抹 霜 工作 . [分隔符] neutral
         2.我们 的 一个 号码 会 非常 详细 地 执行 你 的 指示 [分隔符] 我 团队 的 一个 成员 将 非常 精确 地 执行 你 的 命令  [分隔符] entailment
        
        原始的XNLI覆盖15种语言（含低资源语言）。我们选取其中的中文，并将做格式转换，使得非常容易进入训练和测试阶段。


##### 3.TNEWS 今日头条中文新闻（短文本）分类 Short Text Classificaiton for News

        数据量：训练集(266,000)，验证集(57,000)，测试集(57,000)
        例子：
        6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
        每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code，分类名称，新闻字符串（仅含标题），新闻关键词

##### 4.INEWS 互联网情感分析任务 Sentiment Analysis for Internet News

        数据量：训练集(5,356)，验证集(1,000)，测试集(1,000)     
        例子：
        1_!_00005a3efe934a19adc0b69b05faeae7_!_九江办好人民满意教育_!_近3年来，九江市紧紧围绕“人本教育、公平教育、优质教育、幸福教育”的目标，努力办好人民满意教育，促进了义务教育均衡发展，农村贫困地区办学条件改善。目前，该市特色教育学校有70所 ......
        每行为一条数据，以_!_分割的个字段，从前往后分别是情感类别，数据id，新闻标题，新闻内容

##### 5.DRCD 繁体阅读理解任务 Reading Comprehension for Traditional Chinese
台達閱讀理解資料集 Delta Reading Comprehension Dataset (DRCD)(https://github.com/DRCKnowledgeTeam/DRCD) 屬於通用領域繁體中文機器閱讀理解資料集。 本資料集期望成為適用於遷移學習之標準中文閱讀理解資料集。  

```
数据量：训练集(8,016个段落，26,936个问题)，验证集(1,000个段落，3,524个问题)，测试集(1,000个段落，3,493个问题)  
例子：
{
  "version": "1.3",
  "data": [
    {
      "title": "基督新教",
      "id": "2128",
      "paragraphs": [
        {
          "context": "基督新教與天主教均繼承普世教會歷史上許多傳統教義，如三位一體、聖經作為上帝的啟示、原罪、認罪、最後審判等等，但有別於天主教和東正教，新教在行政上沒有單一組織架構或領導，而且在教義上強調因信稱義、信徒皆祭司， 以聖經作為最高權威，亦因此否定以教宗為首的聖統制、拒絕天主教教條中關於聖傳與聖經具同等地位的教導。新教各宗派間教義不盡相同，但一致認同五個唯獨：唯獨恩典：人的靈魂得拯救唯獨是神的恩典，是上帝送給人的禮物。唯獨信心：人唯獨藉信心接受神的赦罪、拯救。唯獨基督：作為人類的代罪羔羊，耶穌基督是人與上帝之間唯一的調解者。唯獨聖經：唯有聖經是信仰的終極權威。唯獨上帝的榮耀：唯獨上帝配得讚美、榮耀",
          "id": "2128-2",
          "qas": [
            {
              "id": "2128-2-1",
              "question": "新教在教義上強調信徒皆祭司以及什麼樣的理念?",
              "answers": [
                {
                  "id": "1",
                  "text": "因信稱義",
                  "answer_start": 92
                }
              ]
            },
            {
              "id": "2128-2-2",
              "question": "哪本經典為新教的最高權威?",
              "answers": [
                {
                  "id": "1",
                  "text": "聖經",
                  "answer_start": 105
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```
数据格式和squad相同，如果使用简体中文模型进行评测的时候可以将其繁转简(本项目已提供)
        
##### 6.CMRC2018 简体中文阅读理解任务 Reading Comprehension for Simplified Chinese

https://hfl-rc.github.io/cmrc2018/

```
数据量：训练集(短文数2,403，问题数10,142)，试验集(短文数256，问题数1,002)，开发集(短文数848，问题数3,219)  
例子：
{
  "version": "1.0",
  "data": [
    {
        "title": "傻钱策略",
        "context_id": "TRIAL_0",
        "context_text": "工商协进会报告，12月消费者信心上升到78.1，明显高于11月的72。另据《华尔街日报》报道，2013年是1995年以来美国股市表现最好的一年。这一年里，投资美国股市的明智做法是追着“傻钱”跑。所谓的“傻钱”策略，其实就是买入并持有美国股票这样的普通组合。这个策略要比对冲基金和其它专业投资者使用的更为复杂的投资方法效果好得多。",
        "qas":[
                {
                "query_id": "TRIAL_0_QUERY_0",
                "query_text": "什么是傻钱策略？",
                "answers": [
                     "所谓的“傻钱”策略，其实就是买入并持有美国股票这样的普通组合",
                     "其实就是买入并持有美国股票这样的普通组合",
                     "买入并持有美国股票这样的普通组合"
                    ]
                },
                {
                "query_id": "TRIAL_0_QUERY_1",
                "query_text": "12月的消费者信心指数是多少？",
                "answers": [
                    "78.1",
                    "78.1",
                    "78.1"
                    ]
                },
                {
                "query_id": "TRIAL_0_QUERY_2",
                "query_text": "消费者信心指数由什么机构发布？",
                "answers": [
                    "工商协进会",
                    "工商协进会",
                    "工商协进会"
                    ]
                }
            ]
        }
    ]
}
```
数据格式和squad相同

##### 7. BQ 智能客服问句匹配 Question Matching for Customer Service
该数据集是自动问答系统语料，共有120,000对句子对，并标注了句子对相似度值，取值为0或1（0表示不相似，1表示相似）。数据中存在错别字、语法不规范等问题，但更加贴近工业场景。

        数据量：训练集(100,000)，验证集(10,000)，测试集(10,000)
        例子： 
         1.我存钱还不扣的 [分隔符] 借了每天都要还利息吗 [分隔符] 0
         2.为什么我的还没有额度 [分隔符] 为啥没有额度！！ [分隔符] 1

##### 8. MSRANER 命名实体识别 Name Entity Recognition
该数据集共有5万多条中文命名实体识别标注数据（包括人名、地名、组织名），分别用nr、ns、nt表示，其他实体用o表示。

        数据量：训练集(46,364)，测试集(4,365)
        例子： 
         1.据说/o 应/o 老友/o 之/o 邀/o ，/o 梁实秋/nr 还/o 坐/o 着/o 滑竿/o 来/o 此/o 品/o 过/o 玉峰/ns 茶/o 。/o
         2.他/o 每年/o 还/o 为/o 河北农业大学/nt 扶助/o 多/o 名/o 贫困/o 学生/o 。/o

##### 9. THUCNEWS 长文本分类 Long Text classification
该数据集共有4万多条中文新闻长文本标注数据，共14个类别: "体育":0, "娱乐":1, "家居":2, "彩票":3, "房产":4, "教育":5, "时尚":6, "时政":7, "星座":8, "游戏":9, "社会":10, "科技":11, "股票":12, "财经":13。

        数据量：训练集(33,437)，验证集(4,180)，测试集(4,180)
        例子： 
     11_!_科技_!_493337.txt_!_爱国者A-Touch MK3533高清播放器试用　　爱国者MP5简介:　　"爱国者"北京华旗资讯，作为国内知名数码产品制>造商。1993年创立于北京中关村，是一家致力于......
     每行为一条数据，以_!_分割的个字段，从前往后分别是 类别ID，类别名称，文本ID，文本内容。

##### 10.iFLYTEK 长文本分类 Long Text classification

该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。

```
    数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)
    例子： 
17_!_休闲益智_!_玩家需控制一只酷似神龙大侠的熊猫人在科技感十足的未来城市中穿越打拼。感觉很山寨功夫熊猫，自由度非常高，可以做很多你想做的事情......
每行为一条数据，以_!_分割字段，从前往后分别是 类别ID，类别名称，文本内容。
```

##### 11.CHID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test

https://arxiv.org/abs/1906.01265  
成语完形填空，文中多处成语被mask，候选项中包含了近义的成语。

```
    数据量：训练集(84,709)，验证集(3,218)，测试集(3,231)
    例子：
    {
      "content": [
        # 文段0
        "……在热火22年的历史中，他们已经100次让对手得分在80以下，他们在这100次中都取得了胜利，今天他们希望能#idiom000378#再进一步。", 
        # 文段1
        "在轻舟发展过程之中，是和业内众多企业那样走相似的发展模式，去#idiom000379#？还是迎难而上，另走一条与众不同之路。诚然，#idiom000380#远比随大流更辛苦，更磨难，更充满风险。但是有一条道理却是显而易见的：那就是水往低处流，随波逐流，永远都只会越走越低。只有创新，只有发展科技，才能强大自己。", 
        # 文段2
        "最近十年间，虚拟货币的发展可谓#idiom000381#。美国著名经济学家林顿·拉鲁什曾预言：到2050年，基于网络的虚拟货币将在某种程度上得到官方承认，成为能够流通的货币。现在看来，这一断言似乎还嫌过于保守……", 
        # 文段3
        "“平时很少能看到这么多老照片，这次图片展把新旧照片对比展示，令人印象深刻。”现场一位参观者对笔者表示，大多数生活在北京的人都能感受到这个城市#idiom000382#的变化，但很少有人能具体说出这些变化，这次的图片展按照区域发展划分，展示了丰富的信息，让人形象感受到了60年来北京的变化和发展。", 
        # 文段4
        "从今天大盘的走势看，市场的热点在反复的炒作之中，概念股的炒作#idiom000383#，权重股走势较为稳健，大盘今日早盘的震荡可以看作是多头关前的蓄势行为。对于后市，大盘今日蓄势震荡后，明日将会在权重和题材股的带领下亮剑冲关。再创反弹新高无悬念。", 
        # 文段5
        "……其中，更有某纸媒借尤小刚之口指出“根据广电总局的这项要求，2009年的荧屏将很难出现#idiom000384#的情况，很多已经制作好的非主旋律题材电视剧想在卫视的黄金时段播出，只能等到2010年了……"],
      "candidates": [
        "百尺竿头", 
        "随波逐流", 
        "方兴未艾", 
        "身体力行", 
        "一日千里", 
        "三十而立", 
        "逆水行舟", 
        "日新月异", 
        "百花齐放", 
        "沧海一粟"
      ]
    }
```


##### 12. 更多数据集添加中，Comming soon!

更多数据集添加中，如果你有定义良好的数据集，请与我们取得联系。

##### 数据集下载 <a href="https://storage.googleapis.com/chineseglue/chineseGLUEdatasets.v0.0.1.zip">整体下载</a>

或使用命令：

    wget https://storage.googleapis.com/chineseglue/chineseGLUEdatasets.v0.0.1.zip

中文任务基准测评(ChineseGLUE)-排行榜-各任务对比 Evaluation of Dataset for Different Models
---------------------------------------------------------------------

#### TNEWS 短文本分类 Short Text Classificaiton for News (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xlarge | 88.30  | 88.30  |batch_size=32, length=128, epoch=3 |
| BERT-base | 89.80 | 89.78 | batch_size=32, length=128, epoch=3 |
| BERT-wwm-ext-base | 89.88 | 89.81 | batch_size=32, length=128, epoch=3 |
| ERNIE-base  | 89.77 |89.83 | batch_size=32, length=128, epoch=3 |
| RoBERTa-large | 90.00 | 89.91 | batch_size=16, length=128, epoch=3 |
| XLNet-mid |86.14 | 86.26 |  batch_size=32, length=128, epoch=3 |
| RoBERTa-wwm-ext | 89.82 | 89.79 | batch_size=32, length=128, epoch=3 |
| RoBERTa-wwm-large-ext | ***90.05*** | ***90.11*** | batch_size=16, length=128, epoch=3 |

#### XNLI 自然语言推理  Natural Language Inference (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xlarge | 74.0? | 74.0? |batch_size=64, length=128, epoch=2 |
| BERT-base | 77.80 | 77.80 | batch_size=64, length=128, epoch=2 |
| BERT-wwm-ext-base | 79.4 | 78.7 | batch_size=64, length=128, epoch=2 |
| ERNIE-base  | 79.7  |78.6 | batch_size=64, length=128, epoch=2 |
| RoBERTa-large |***80.2*** |79.9 | batch_size=64, length=128, epoch=2 |
| XLNet-mid | 79.2 | 78.7 | batch_size=64, length=128, epoch=2 |
| RoBERTa-wwm-ext | 79.56 | 79.28 | batch_size=64, length=128, epoch=2 |
| RoBERTa-wwm-large-ext | ***80.20*** | ***80.04*** | batch_size=16, length=128, epoch=2 |

注：ALBERT-xlarge，在XNLI任务上训练暂时还存在有问题

#### LCQMC 口语化描述的语义相似度匹配 Semantic Similarity Task (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xlarge | 89.00  | 86.76  |batch_size=64, length=128, epoch=3 |
| BERT-base | 89.4  | 86.9  | batch_size=64, length=128, epoch=3 |
| BERT-wwm-ext-base |89.1   | ***87.3*** |  batch_size=64, length=128, epoch=3 |
| ERNIE-base  | 89.8  | 87.2 | batch_size=64, length=128, epoch=3|
| RoBERTa-large |***89.9***  | 87.2|  batch_size=64, length=128, epoch=3 |
| XLNet-mid | 86.14 | 85.98 | batch_size=64, length=128, epoch=3 |
| RoBERTa-wwm-ext | 89.08 | 86.33 | batch_size=64, length=128, epoch=3 |
| RoBERTa-wwm-large-ext | 89.79 | 86.82 | batch_size=16, length=128, epoch=3 |

#### INEWS 互联网情感分析 Sentiment Analysis for Internet News (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xlarge | 81.80 | 82.40 |batch_size=32, length=512, epoch=8 |
| BERT-base | 81.29 | 82.70 | batch_size=16, length=512, epoch=3 |
| BERT-wwm-ext-base | 81.93 | 83.46 | batch_size=16, length=512, epoch=3 |
| ERNIE-base  | ***84.50*** |***85.14*** | batch_size=16, length=512, epoch=3 |
| RoBERTa-large | 81.90 | 84.00 | batch_size=4, length=512, epoch=3 |
| XLNet-mid | 82.00 | 84.00 | batch_size=8, length=512, epoch=3 |
| RoBERTa-wwm-ext | 82.98 | 82.28 | batch_size=16, length=512, epoch=3 |
| RoBERTa-wwm-large-ext | 83.73 | 82.78 | batch_size=4, length=512, epoch=3 |

#### DRCD 繁体阅读理解 Reading Comprehension for Traditional Chinese (F1, EM)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base |F1:92.30 EM:86.60 | F1:91.46 EM:85.49 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
| BERT-wwm-ext-base |F1:93.27 EM:88.00 | F1:92.63 EM:87.15 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
| ERNIE-base  |F1:92.78 EM:86.85 | F1:92.01 EM:86.03 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
| ALBERT-large  |F1:93.90 EM:88.88 | F1:93.06 EM:87.52 |  batch=32, length=512, epoch=3 lr=2e-5 warmup=0.05 |
| ALBERT-xlarge |F1:94.63 EM:89.68 | F1:94.70 EM:89.78 |  batch_size=32, length=512, epoch=3 lr=2.5e-5 warmup=0.06 |
| ALBERT-tiny |F1:81.51 EM:71.61 | F1:80.67 EM:70.08 |  batch=32, length=512, epoch=3 lr=2e-4 warmup=0.1 |
| RoBERTa-large |F1:94.93 EM:90.11 | F1:94.25 EM:89.35 |  batch=32, length=256, epoch=2 lr=3e-5 warmup=0.1|
| xlnet-mid |F1:92.08 EM:84.40 | F1:91.44 EM:83.28 | batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1 |
| RoBERTa-wwm-ext |F1:94.26 EM:89.29 | F1:93.53 EM:88.12 |  batch=32, length=512, epoch=2 lr=3e-5 warmup=0.1|
| RoBERTa-wwm-large-ext |***F1:95.32 EM:90.54*** | ***F1:95.06 EM:90.70*** | batch=32, length=512, epoch=2 lr=2.5e-5 warmup=0.1 |


#### CMRC2018 阅读理解 Reading Comprehension for Simplified Chinese (F1, EM)：

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

#### CHID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) |  训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base	| 82.2 | 82.04 | batch=24, length=64, epoch=3 lr=2e-5 |
| BERT-wwm-ext-base	|- |-|	- |
| ERNIE-base	|- | - | - |
| ALBERT-large	|- | - | - |
| ALBERT-xlarge	|- | - | - |
| ALBERT-tiny	|- | - | - |
| RoBERTa-large	| 85.31 | 84.5 | batch=24, length=64, epoch=3 lr=2e-5  |
| xlnet-mid	|- | - | - |
| RoBERTa-wwm-ext	|83.78 | 83.62 | batch=24, length=64, epoch=3 lr=2e-5  |
| RoBERTa-wwm-large-ext	|***85.81*** | ***85.37*** | batch=24, length=64, epoch=3 lr=2e-5  |


#### BQ 智能客服问句匹配 Question Matching for Customer Service (Accuracy)：

| 模型 | 开发集（dev） | 测试集（test） | 训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base | 85.86 | 85.08 | batch_size=64, length=128, epoch=3 |
| BERT-wwm-ext-base | 86.05 | ***85.21*** |batch_size=64, length=128, epoch=3 |
| ERNIE-base | 85.92 | 84.47 | batch_size=64, length=128, epoch=3 |
| RoBERTa-large | 85.68 | 85.20 | batch_size=8, length=128, epoch=3 |
| XLNet-mid | 79.81 | 77.85 | batch_size=32, length=128, epoch=3 |
| ALBERT-xlarge | 85.21 | 84.21 | batch_size=16, length=128, epoch=3 |
| ALBERT-tiny | 82.04 | 80.76 | batch_size=64, length=128, epoch=5 |
| RoBERTa-wwm-ext | 85.31 | 84.02 | batch_size=64, length=128, epoch=3 |
| RoBERTa-wwm-large-ext | ***86.34*** | 84.90 | batch_size=16, length=128, epoch=3 |

#### MSRANER 命名实体识别 Name Entity Recognition (F1):

| 模型 | 测试集（test） | 训练参数 |
| :----: | :----: | :----: |
| BERT-base | 95.38 | batch_size=16, length=256, epoch=5, lr=2e-5 |
| BERT-wwm-ext-base | 95.26 | batch_size=16, length=256, epoch=5, lr=2e-5 |
| ERNIE-base | 95.17 | batch_size=16, length=256, epoch=5, lr=2e-5 |
| RoBERTa-large | ***96.07*** | batch_size=8, length=256, epoch=5, lr=2e-5 |
| XLNet-mid | - | - |
| ALBERT-xlarge | 89.51 | batch_size=16, length=256, epoch=8, lr=7e-5 |
| ALBERT-base | 92.47 | batch_size=32, length=256, epoch=8, lr=5e-5 |
| ALBERT-tiny | 84.77 | batch_size=32, length=256, epoch=8, lr=5e-5 |
| RoBERTa-wwm-ext | 95.06 | batch_size=16, length=256, epoch=5, lr=2e-5 |
| RoBERTa-wwm-large-ext | 95.32 | batch_size=8, length=256, epoch=5, lr=2e-5 |

#### THUCNEWS 长文本分类 Long Text Classification (Accuracy)：

| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xlarge | 95.74  | 95.45 |batch_size=32, length=512, epoch=8 |
| ALBERT-tiny | 92.63 | 93.54 | batch_size=64, length=128, epoch=5 |
| BERT-base | 95.28 | 95.35 | batch_size=8, length=128, epoch=3 |
| BERT-wwm-ext-base | 95.38 | 95.57 | batch_size=8, length=128, epoch=3 |
| ERNIE-base  | 94.35 | 94.90 | batch_size=16, length=256, epoch=3 |
| RoBERTa-large | 94.52 | 94.56 | batch_size=2, length=256, epoch=3 |
| XLNet-mid | 94.04 | 94.54 | batch_size=16, length=128, epoch=3 |
| RoBERTa-wwm-ext | 95.59 | 95.52 | batch_size=16, length=256, epoch=3 |
| RoBERTa-wwm-large-ext | ***96.10*** | ***95.93*** | batch_size=32, length=512, epoch=8 |

#### iFLYTEKData 长文本分类 Long Text Classification (Accuracy)：

|         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
| :-------------------: | :----------: | :-----------: | :--------------------------------: |
|     ALBERT-xlarge     |    61.94     |     61.34     | batch_size=32, length=128, epoch=3 |
|      ALBERT-tiny      |    44.83     |     44.62     | batch_size=32, length=256, epoch=3 |
|       BERT-base       |    63.57     |     63.48     | batch_size=32, length=128, epoch=3 |
|   BERT-wwm-ext-base   |    63.83     |     63.75     | batch_size=32, length=128, epoch=3 |
|      ERNIE-base       |    61.75     |     61.80     | batch_size=24, length=256, epoch=3 |
|     RoBERTa-large     |    63.80     |     63.91     | batch_size=32, length=128, epoch=3 |
|       XLNet-mid       |    60.16     |     60.04     | batch_size=16, length=128, epoch=3 |
|    RoBERTa-wwm-ext    |    64.18     |       -       | batch_size=16, length=128, epoch=3 |
| RoBERTa-wwm-large-ext | ***65.19***  |  ***65.10***  | batch_size=32, length=128, epoch=3 |

基线模型-代码 Start Codes for Baselines 
---------------------------------------------------------------------

我们为您提供了可以“一键运行”的脚本来辅助您更快的在指定模型上运行特定任务。

以在 Bert 模型上运行“BQ 智能客服问句匹配”任务为例，您可以直接在 chineseGLUE/baselines/models/**bert**/ 下运行 run_classifier_**bq**.sh 脚本。

  ```bash
  cd chineseGLUE/baselines/models/bert/
  sh run_classifier_bq.sh
  ```


该脚本将会自动下载“BQ 智能客服问句匹配”数据集（保存在chineseGLUE/baselines/glue/chineseGLUEdatasets/**bq**/ 文件夹下）和Bert模型（保存在 chineseGLUE/baselines/models/bert/prev_trained_model/ 下）。

<!--1. 数据集整体下载，解压到glue文件夹里  -->
<!--  ```cd glue ```  -->
<!--  ```wget https://storage.googleapis.com/chineseglue/chineseGLUEdatasets.v0.0.1.zip```-->
<!--   注：lcqmc数据集，请从<a href="http://icrc.hitsz.edu.cn/info/1037/1146.htm">这里</a>申请或搜索网络-->

<!--2. 训练模型  -->

<!--    ```a.将预训练模型下载解压到对应的模型中prev_trained_model文件夹里。``` -->
<!--         ```以bert和albert为例子：```-->
            
<!--         ``` a1. albert  ``` -->
<!--         ```https://github.com/brightmart/albert_zh ```  -->
<!--         ```a1. bert  ``` -->
<!--         ```https://github.com/google-research/bert ```    -->
    
<!--     ```b.修改run_classifier.sh指定模型路径  ``` -->
    
<!--     ```c.运行各个模型文件夹下的run_classifier.sh即可 ```  -->
<!--       ```sh run_classifier.sh```-->
具体内容详见：<a href="https://github.com/chineseGLUE/chineseGLUE/tree/master/baselines">基准模型-模型训练</a>

#### 开放测评提交入口：<a href="http://106.13.187.75:8003/">我要提交</a>

<img src="https://github.com/chineseGLUE/chineseGLUE/blob/master/resources/img/chineseGLUE_landing.jpeg"  width="80%" height="40%" />


语料库：语言建模、预训练或生成型任务 Corpus for Langauge Modelling, Pre-training, Generating tasks
---------------------------------------------------------------------
可用于语言建模、预训练或生成型任务等，数据量超过10G，主要部分来自于<a href="https://github.com/brightmart/nlp_chinese_corpus">nlp_chinese_corpus项目</a>

当前语料库按照【预训练格式】处理，内含有多个文件夹；每个文件夹有许多不超过4M大小的小文件，文件格式符合预训练格式：每句话一行，文档间空行隔开。

包含如下子语料库（总共14G语料）：

1、新闻语料: 8G语料，分成两个上下两部分，总共有2000个小文件。

2、社区互动语料：3G语料，包含3G文本，总共有900多个小文件。

3、维基百科：1.1G左右文本，包含300左右小文件。

4、评论数据：2.3G左右文本，含有811个小文件，合并<a href="https://github.com/InsaneLife/ChineseNLPCorpus">ChineseNLPCorpus</a>的多个评论数据，清洗、格式转换、拆分成小文件。

这些语料，你可以通过上面这两个项目，清洗数据并做格式转换获得；

你也可以通过邮件申请（chineseGLUE#163.com）获得单个项目的语料，告知单位或学校、姓名、语料用途；

如需获得ChineseGLUE项目下的所有语料，需成为ChineseGLUE组织成员，并完成一个（小）任务。

成为ChineseGLUE组织的创始成员 Members
---------------------------------------------------------------------
##### 你将可以 Benefits：

1、成功中国第一个中文任务基准测评的创始会员

2、能与其他专业人士共同贡献力量，促进中文自然语言处理事业的发展

3、参与部分工作后，获得已经清洗并预训练的后的、与英文wiki & bookCorpus同等量级、大规模的预训练语料，用于研究目的。

4、优先使用state of the art的中文预训练模型，包括各种体验版或未公开版本

##### 如何参与 How to join with us：

发送邮件 chineseGLUE#163.com，简要介绍你自己、背景、工作或研究方向、你的组织、在哪方面可以为社区贡献力量，我们评估后会与你取得联系你。

任务清单 TODO LIST
---------------------------------------------------------------------
1、搜集、挖掘1个有代表性的数据集，一般为分类或句子对任务 (需要额外5个数据集)

2、阅读理解任务转化成句子对任务（如线索与问题或答案），并做测评，数据应拆分成训练、验证和测试集。

3、基线模型baselises在特定任务模型的训练、预测的方法和脚本(支持PyTorch、Keras)；

4、对当前主流模型（如bert/bert_wwm_ext/roberta/albert/ernie/ernie2.0等），结合ChineseGLUE的数据集，做准确率测试。

   如： XLNet-mid在LCQMC数据集上做测试

5、是否还有没有参与测评的模型？

##### 其他
6、排行榜landing页

7、介绍中文语言理解测评基准(ChineseGLUE)

8、测评系统主要功能开发

Timeline 时间计划:
---------------------------------------------------------------------
2019-10-20 to 2019-12-31: beta version of ChineseGLUE

2020.1.1 to 2020-12-31: official version of ChineseGLUE

2021.1.1 to 2021-12-31: super version of ChineseGLUE

Contribution 贡献你的力量，从今天开始
---------------------------------------------------------------------

Share your data set with community or make a contribution today! Just send email to chineseGLUE#163.com, 

or join QQ group: 836811304

中文基准测评成员 Members
---------------------------------------------------------------------
#### 顾问 Adviser：
张俊林，中国中文信息学会理事，中科院软件所博士，新浪微博机器学习团队AI Lab负责人。技术书籍《这就是搜索引擎：核心技术详解》（该书荣获全国第十二届优秀图书奖）、《大数据日知录：架构与算法》的作者。

崔一鸣，哈工大讯飞联合实验室（HFL）资深级研究员。中文机器阅读理解CMRC系列评测发起者，负责多个中文预训练模型项目，所领导的团队多次在SQuAD、CoQA、QuAC、HotpotQA等国际阅读理解评测中荣登榜首。

#### 创始会员 Charter Members（排名不分先后）：
徐亮，<a href='https://github.com/brightmart'>brightmart</a>，中文任务基准测评ChineseGLUE发起人。杭州实在智能算法专家，多个预训练模型中文版、文本分类开源项目作者。

Danny Lan，CMU博士、google AI 研究员，SOTA语言理解模型AlBERT第一作者。

徐国强，MIT博士，平安集团上海Gammalab负责人。

张轩玮，毕业于北京大学，目前在爱奇艺从事nlp有关的工作，之前做过热点聚合，文本分类，标签生成，机器翻译方面的工作。

谢炜坚，百度大数据部，算法工程师，从事NLP相关工作，包括任务驱动型对话、检索式问答、语义匹配、文本分类、情感分析等工作。

曹辰捷，平安金融壹账通，算法工程师，负责阅读理解和预训练相关业务，CMRC2019阅读理解冠军团队成员。

喻聪，来自杭州实在智能，主要研究多轮对话，意图识别，实体抽取，知识问答相关任务。

谢恩宁，大搜车，围绕汽车领域语对话机器人，负责NLU部分。

李露，来自华中师范大学计算机学院，曾参与某项目筹备中文自然语言推理的数据集；暑期在平安科技实习，主要负责利用自然语言处理最新模型进行序列标注和情感分类任务。

董倩倩，来自中科院自动化所，phd在读，主要研究语音翻译，曾参与多个中文NLP项目。

王荣钊，北京大学数学科学学院读研究生，目前在微软NLP Group参与实习。做过一些NLU的相关任务，熟悉常见的模型，训练过一些LU的任务以及一些机器翻译的任务，对pytorch比较熟悉。

田垠，毕业于浙江大学，杭州实在智能算法工程师，方向主要为基于知识图谱的推理引擎、文字检测。

刘伟棠，大华，albert_pytorch项目作者。

陈哲乾，浙江大学计算机学院博士，一知智能联合创始人，2017年代表一知智能参加斯坦福大学的举办SQuAD机器阅读理解比赛，获得单模型组世界第二、多模型组世界第三的优异成绩。主导设计一知智能大脑项目。

叶琛，浙江大学研究生，一知智能算法实习生。目前在做模型预训练 & 蒸馏、阅读理解方面工作。

更多创始会员，陆续添加中。。。

#### 志愿者 Volunteers ：
许皓天，清华电子系毕业，目前在阿里cro线，负责模型蒸馏、领域自适应、相似检索、多语言迁移、弱监督学习等相关工作。

胡锦毅，清华大学计算机系，大三，在清华大学自然语言处理与社会人文计算实验室研究学习，导师是孙茂松教授；“九歌”人工智能诗歌创作系统2.0，获CCL2019最佳系统展示奖。

盛泳潘，电子科技大学博士（即将毕业），后续将尝试用中文领域的data做知识图谱构建以及语义依存分析等问题。

杜则尧，GPT2-Chinese作者。

更多志愿者，陆续添加中。。。

#### Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)


Reference:
---------------------------------------------------------------------
1、<a href="https://openreview.net/pdf?id=rJ4km2R5t7">GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding</a>

2、<a href="https://w4ngatang.github.io/static/papers/superglue.pdf">SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems</a>

3、<a href="https://www.aclweb.org/anthology/C18-1166.pdf">LCQMC: A Large-scale Chinese Question Matching Corpus</a>

4、<a href="https://arxiv.org/pdf/1809.05053.pdf">XNLI: Evaluating Cross-lingual Sentence Representations</a>

5、<a href="https://github.com/fate233/toutiao-text-classfication-dataset">TNES: toutiao-text-classfication-dataset</a>

6、<a href="https://github.com/brightmart/nlp_chinese_corpus">nlp_chinese_corpus: 大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP</a>

7、<a href="https://github.com/InsaneLife/ChineseNLPCorpus">ChineseNLPCorpus</a>

8、<a href="https://arxiv.org/abs/1909.11942">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

9、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

10、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>

