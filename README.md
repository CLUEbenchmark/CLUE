# CLUE benchmark
Language Understanding Evaluation benchmark for Chinese: datasets, baselines, pre-trained models, corpus and leaderboard

中文语言理解测评基准，包括代表性的数据集、基准(预训练)模型、语料库、排行榜。  

我们会选择一系列有一定代表性的任务对应的数据集，做为我们测试基准的数据集。这些数据集会覆盖不同的任务、数据量、任务难度。

中文任务基准测评(CLUE benchmark)-排行榜 Leaderboard
---------------------------------------------------------------------
#####  排行榜会定期更新                     数据来源: https://github.com/CLUEbenchmark/CLUE

#### 分类任务(v1版本,正式版)

| 模型   | Score  | 参数    | AFQMC  | TNEWS'  | IFLYTEK'   | CMNLI   | COPA | WSC | CSL  |
| :----:| :----: | :----: | :----: |:----: |:----: |:----: |:----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>        | 69.70% | 108M |  73.70% | 56.58%  | 60.29% |   | 57.40% | 62.0% | 80.36% |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a>      | 70.47% | 108M  | 74.07% | 56.84%  | 59.43% | | 61.40%  | 61.1%  | 80.63% |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>         | 70.55% | 108M  | 73.83% | 58.33% | 58.96% |  | **65.00%**  | 60.8%  | 79.1%      |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a>      | 72.63% | 334M  | 74.02% | 57.86%  | 62.55% | | 61.40% | 72.7%   | 81.36%       |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>  | 68.65% | 200M | 70.50% | 56.24% | 57.85% | | 53.80%   | 64.4%   | 81.26%     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a>      | - | 59M   | -  | - | - | - | - | - | -  | -  | -     |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a>        | 61.92% | 4M | 69.92% | 53.35% | 36.18% |  | 49.80%  | 58.5%   | 74.56% |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>   | 71.72% | 108M  | 74.04% | 56.94% | 60.31% | | 63.60%  | 67.8% | 81.0% |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | **73.45%** | 330M | **76.55%** | **58.61%** | **62.98%** |  | 59.40% | **74.6%** | **82.13%** |


    注：AFQMC:蚂蚁语义相似度(Acc)；TNEWS:文本分类(Acc)；IFLYTEK:长文本分类(Acc); CMNLI: 自然语言推理中文版; 
       COPA: 因果推断; WSC: Winograd模式挑战中文版; CSL: 中国科学文献数据集; Score总分是通过计算1-9数据集得分平均值获得；
      '代表对原数据集使用albert_tiny模型筛选后获得，数据集与原数据集不同,从而可能导致在这些数据集上albert_tiny表现略低.

#### 阅读理解任务

| 模型 | Score | 参数 | DRCD | CMRC2018 | CHID |
| :----:| :----: | :----: | :----: |:----: |:----: |
| <a href="https://github.com/google-research/bert">BERT-base</a>	| 79.08 | 108M | 85.49 	| 69.72 | 82.04 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">BERT-wwm-ext</a> | 81.09 | 108M | 87.15 | 73.23 | 82.90 |
| <a href="https://github.com/PaddlePaddle/ERNIE">ERNIE-base</a>	| 80.54 | 108M | 86.03 | 73.32 | 82.28 |
| <a href="https://github.com/brightmart/roberta_zh">RoBERTa-large</a> | 83.32 | 334M 	| 89.35 | 76.11 | 84.50 |
| <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">XLNet-mid</a>	| 77.75 | 209M | 83.28 | 66.51  | 83.47 |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xlarge</a> | 81.52 | 59M | 89.78 | 75.22 | 79.55 |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-xxlarge</a> | - | - | - | - | - |
| <a href="https://github.com/brightmart/albert_zh">ALBERT-tiny</a> | 55.73 | 1.8M | 70.08 | 53.68 | 43.53 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-ext</a>  | 81.88 | 108M  | 88.12 | 73.89 | 83.62 |
| <a href="https://github.com/ymcui/Chinese-BERT-wwm">RoBERTa-wwm-large</a> | ***84.22*** | 330M |	***90.70*** |	***76.58*** | ***85.37*** |

DRCD、CMRC2018: 繁体、简体抽取式阅读理解(F1, EM)；CHID：成语多分类阅读理解(Acc)；

注：阅读理解上述指标中F1和EM共存的情况下，取EM为最终指标

一键运行.基线模型与代码 Baseline with codes
---------------------------------------------------------------------
    使用方式：
    1、克隆项目 
       git clone https://github.com/CLUEbenchmark/CLUE.git
    2、进入到相应的目录
       分类任务：
           cd CLUE/baselines/models/bert  
       或阅读理解任务：
           cd CLUE/baselines/models_pytorch/mrc_pytorch
    3、运行对应任务的脚本: 会自动下载模型和任务数据并开始运行。
       bash run_classifier_xxx.sh
       如运行 bash run_classifier_iflytek.sh 会开始iflytek任务的训练

### 生成提交文件

    分类任务:
    在run_classifier_xxx.sh中设置do_predict为true（如果不要再训练或者评估可以将do_train, do_eval置为false)
    运行即可得到相应的提交文件json格式结果 
   或见<a href="https://github.com/CLUEbenchmark/CLUE/blob/master/baselines/models/bert/run_classifier.py#L932-L951">代码实现</a>

    阅读理解任务:

     TODO
    ​    

数据集下载见本项目最后部分 

CLUE benchmark的定位 Vision
---------------------------------------------------------------------
为更好的服务中文语言理解、任务和产业界，做为通用语言模型测评的补充，通过完善中文语言理解基础设施的方式来促进中文语言模型的发展

*** 2019-10-13: 新增评测官网入口; INEWS基线模型 ***

  <a href="http://106.13.187.75:8003/"> 评测入口</a>

数据集介绍与下载 Introduction of datasets 
--------------------------------------------------------------------

 <a href="https://storage.googleapis.com/cluebenchmark/tasks/clue_submit_examples.zip">提交样例下载</a>

##### 1. AFQMC 蚂蚁金融语义相似度 Ant Financial  Question Matching Corpus

        数据量：训练集（34334）验证集（4316）测试集（3861）
        例子：14870	蚂蚁借呗等额还款可以换成先息后本吗	借呗有先息到期还本吗	0
        每行为一条数据， 以\tab分割的4个字段，从前往后分别是 ID, 第一个句子，第二个句子，类别

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/afqmc_public.zip" > AFQMC'数据集下载</a>

##### 2.TNEWS' 今日头条中文新闻（短文本）分类 Short Text Classificaiton for News

        数据量：训练集(266,000)，验证集(57,000)，测试集(57,000)
        例子：
        6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
        每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code，分类名称，新闻字符串（仅含标题），新闻关键词

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/tnews_public.zip" > TNEWS'数据集下载</a>

##### 3.IFLYTEK' 长文本分类 Long Text classification
该数据集共有1.7万多条关于app应用描述的长文本标注数据，包含和日常生活相关的各类应用主题，共119个类别："打车":0,"地图导航":1,"免费WIFI":2,"租车":3,….,"女性":115,"经营":116,"收款":117,"其他":118(分别用0-118表示)。
```
    数据量：训练集(12,133)，验证集(2,599)，测试集(2,600)
    例子： 
17_!_休闲益智_!_玩家需控制一只酷似神龙大侠的熊猫人在科技感十足的未来城市中穿越打拼。感觉很山寨功夫熊猫，自由度非常高，可以做很多你想做的事情......
每行为一条数据，以_!_分割字段，从前往后分别是 类别ID，类别名称，文本内容。
```
   <a href="https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip" > IFLYTEK'数据集下载</a>

##### 4.CMNLI 语言推理任务 Chinese Multi-Genre NLI

CMNLI数据由两部分组成：XNLI和MNLI。数据来自于fiction，telephone，travel，government，slate等，对原始MNLI数据和XNLI数据进行了中英文转化，合并两部分数据并打乱顺序后，重新划分训练、验证和测试集。该数据集可用于判断给定的两个句子之间属于蕴涵、中立、矛盾关系。

```
    数据量：train(391,782)，matched(12,426)，mismatched(13,880)
    例子：
    {"sentence1": "新的权利已经足够好了", "sentence2": "每个人都很喜欢最新的福利", "gold_label": "neutral"}
```
   <a href="https://storage.googleapis.com/cluebenchmark/tasks/cmnli_public.zip" > CMNLI数据集下载</a>

##### 5. COPA 因果推断-中文版 Choice of Plausible Alternatives

自然语言推理的数据集，给定一个假设以及一个问题表明是因果还是影响，并从两个选项中选择合适的一个。遵照原数据集，我们使用了acc作为评估标准。

```
    数据量：训练集(400)，验证集(100)，测试集(500)
    例子： 
    1. {"idx": 7, "premise": "那人在杂货店买东西时打折了。", "choice0": "他向收银员打招呼。", "choice1": "他用了一张优惠券。", "question": "cause", "label": 1}
    2. {"idx": 8, "premise": "医师误诊了病人。", "choice0": "该患者对医生提起了医疗事故诉讼。", "choice1": "患者向医生披露了机密信息。", "question": "effect", "label": 0}
其中label的标注，0表示choice0，1 表示choice1。原先的COPA数据集是英文的，我们使用机器翻译以及人工翻译的方法，并做了些微的用法习惯上的调整，并根据中文的习惯进行了标注，得到了这份数据集。
```

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/copa_public.zip" > COPA数据集下载</a>


##### 6. WSC Winograd模式挑战中文版  The Winograd Schema Challenge,Chinese Version
威诺格拉德模式挑战赛是图灵测试的一个变种，旨在判定AI系统的常识推理能力。参与挑战的计算机程序需要回答一种特殊但简易的常识问题：代词消歧问题，即对给定的名词和代词判断是否指代一致。
```
数据量：训练集(532)，验证集(104)，测试集(143) 
例子：
{"target": 
    {"span2_index": 28, 
     "span1_index": 0, 
     "span1_text": "马克", 
     "span2_text": "他"}, 
     "idx": 0, 
     "label": "false", 
     "text": "马克告诉皮特许多关于他自己的谎言，皮特也把这些谎言写进了他的书里。他应该多怀疑。"}
```

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/wsc_public.zip" > WSC数据集下载</a>


##### 7. CSL 论文关键词识别 Keyword Recognition
中文科技文献数据集包含中文核心论文摘要及其关键词。
用tf-idf生成伪造关键词与论文真实关键词混合，生成摘要-关键词对，关键词中包含伪造的则标签为0。
```
    数据量：训练集(20,000)，验证集(3,000)，测试集(3,000)
    例子： 
    通过研究Windows环境下USB设备的工作原理，应用操作系统与USB设备驱动通信获取设备描述和设备ID等信息的机制，提出了一种实用有效的USB设备监控技术。实现了在开机前后两种情况下对USB设备的实时监控，有效地避免了其他监控技术的漏洞。实验结果证明，该方法是可靠有效的。	设备描述 设备ID Windows环境 安全监控	1
    通过研究Windows环境下USB设备的工作原理，应用操作系统与USB设备驱动通信获取设备描述和设备ID等信息的机制，提出了一种实用有效的USB设备监控技术。实现了在开机前后两种情况下对USB设备的实时监控，有效地避免了其他监控技术的漏洞。实验结果证明，该方法是可靠有效的。	设备 技术 安全监控 设备描述	    0
```
   <a href="https://storage.googleapis.com/cluebenchmark/tasks/csl_public.zip" > CSL数据集下载</a>

##### 8.CMRC2018 简体中文阅读理解任务 Reading Comprehension for Simplified Chinese
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

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/cmrc2018_public.zip" > CMRC2018数据集下载</a>


##### 9.DRCD 繁体阅读理解任务 Reading Comprehension for Traditional Chinese
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
   <a href="https://storage.googleapis.com/cluebenchmark/tasks/drcd_public.zip" > DRCD2018数据集下载</a>

##### 10.CHID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test
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

   <a href="https://storage.googleapis.com/cluebenchmark/tasks/chid_public.zip" > CHID数据集下载</a>


##### 更多数据集添加中，Comming soon!
如果你有定义良好的数据集并愿意为社区做贡献，请与我们取得联系 ChineseGLUE#163.com

##### 数据集整体下载 

<a href="#">整体下载 Comining Soon</a>最近几天，会添加中

或使用命令：wget <url>

Data filter method

## 难样本数据集筛选方法

为了增加模型区分度和增大数据集难度，我们采用**k折交叉验证**的方式对v0版本的数据集进行过滤，最终得到v1版本。

```
具体步骤：
1.将特定任务的数据集集中在一起，同时选择一个基准测试模型（如AlbertTiny）
2.将数据集均匀分成k份；每次选择其中1份当验证集，剩下的都作为训练集，训练基准模型并在验证集上测试、保留预测结果
3.重复步骤二k次，让每一份数据都有机会当验证集，过完整个数据集
4.将k份验证集的预测结果合并；保留其中预测错误的样本（可以认为是较难的数据），并删除一部分预测正确的样本。最后重新划分出训练集、验证集、测试集
5.如果希望进一步筛选难样本，重复步骤2-4即可
```

Notes：

```
1.k一般选择4-6
2.难样本，是指在交叉验证过程中模型预测错误的样本，也是我们希望尽可能保留的样本。模型预测正确的样本最终会被优先排除一部分
```

内容体系 Contents
--------------------------------------------------------------------
Language Understanding Evaluation benchmark for Chinese(ChineseGLUE) got ideas from GLUE, which is a collection of 

resources for training, evaluating, and analyzing natural language understanding systems. ChineseGLUE consists of: 

##### 1）中文任务的基准测试，覆盖多个不同程度的语言任务 

A benchmark of several sentence or sentence pair language understanding tasks. 
Currently the datasets used in these tasks are come from public. We will include datasets with private test set before the end of 2019.

##### 2）公开的排行榜 Leaderboard 

A public leaderboard for tracking performance. You will able to submit your prediction files on these tasks, each task will be evaluated and scored, a final score will also be available.

##### 3）基线模型，包含开始的代码、预训练模型  Baselines with code

baselines for ChineseGLUE tasks. baselines will be available in TensorFlow,PyTorch,Keras and PaddlePaddle.

##### 4）语料库，用于语言建模、预训练或生成型任务  Corpus

A huge amount of raw corpus for pre-train or language modeling research purpose. It will contains around 10G raw corpus in 2019; 

In the first half year of 2020, it will include at least 30G raw corpus; By the end of 2020, we will include enough raw corpus, such as 100G, so big enough that you will need no more raw corpus for general purpose language modeling.
You can use it for general purpose or domain adaption, or even for text generating. when you use for domain adaption, you will able to select corpus you are interested in.

##### 5）工具包 toolkit

An easy to use toolkit that can run specific task or model with one line of code. You can easily change configuration, task or model.

##### 6) 技术报告

Techical report with details

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


各任务详细对比
---------------------------------------------------------------------
 Evaluation of Dataset for Different Models

#### AFQMC 蚂蚁语义相似度 Ant Semantic Similarity (Accuracy)：
|         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
| :-------------------: | :----------: | :-----------: | :--------------------------------: |
|     ALBERT-xxlarge     |    -     |     -   | batch_size=16, length=128, epoch=3 |
|      ALBERT-tiny      |    69.13%     |    69.92%    | batch_size=16, length=128, epoch=3 |
|       BERT-base       |    74.16%     |     73.70%  | batch_size=16, length=128, epoch=3 |
|   BERT-wwm-ext-base   |    73.74%     |      74.07%   | batch_size=16, length=128, epoch=3 |
|      ERNIE-base       |         74.88% |      73.83%    | batch_size=16, length=128, epoch=3 |
|     RoBERTa-large     |     73.32%    |       74.02%   | batch_size=16, length=128, epoch=3 |
|       XLNet-mid       |     70.73%    |   70.50%       | batch_size=16, length=128, epoch=3 |
|    RoBERTa-wwm-ext    |   74.30%      |      74.04%       | batch_size=16, length=128, epoch=3 |
| RoBERTa-wwm-large-ext |  74.92% |  76.55% | batch_size=16, length=128, epoch=3 |

#### TNEWS' 头条新闻分类 Toutiao News Classification (Accuracy)：
|         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
| :-------------------: | :----------: | :-----------: | :--------------------------------: |
|     ALBERT-xxlarge     |    -     |         | batch_size=16, length=128, epoch=3 |
|      ALBERT-tiny      |    53.55%     |       53.35%   | batch_size=16, length=128, epoch=3 |
|       BERT-base       |    56.09%     |     56.58%    | batch_size=16, length=128, epoch=3 |
|   BERT-wwm-ext-base   |     56.77%    |    56.86%      | batch_size=16, length=128, epoch=3 |
|      ERNIE-base       |     58.24%    |     58.33%     | batch_size=16, length=128, epoch=3 |
|     RoBERTa-large     |     57.95%    |      57.84%    | batch_size=16, length=128, epoch=3 |
|       XLNet-mid       |    56.09%     |      56.24%    | batch_size=16, length=128, epoch=3 |
|    RoBERTa-wwm-ext    |   57.51%      |      56.94%       | batch_size=16, length=128, epoch=3 |
| RoBERTa-wwm-large-ext |  58.32% | 58.61%  | batch_size=16, length=128, epoch=3 |

#### IFLYTEK' 长文本分类 Long Text Classification (Accuracy)：
|         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
| :-------------------: | :----------: | :-----------: | :--------------------------------: |
|     ALBERT-xlarge     |    -     |     -     | batch_size=32, length=128, epoch=3 |
|      ALBERT-tiny      |    37.54    |     36.18     | batch_size=32, length=128, epoch=3 |
|       BERT-base       |    60.37    |     60.29     | batch_size=32, length=128, epoch=3 |
|   BERT-wwm-ext-base   |    59.88    |     59.43     | batch_size=32, length=128, epoch=3 |
|      ERNIE-base       |    59.52    |     58.96     | batch_size=32, length=128, epoch=3 |
|     RoBERTa-large     |    62.6    |     62.55     | batch_size=24, length=128, epoch=3 |
|       XLNet-mid       |    57.72    |     57.85     | batch_size=32, length=128, epoch=3 |
|    RoBERTa-wwm-ext    |    60.8    |       60.31       | batch_size=32, length=128, epoch=3 |
| RoBERTa-wwm-large-ext | **62.75** |  **62.98**  | batch_size=24, length=128, epoch=3 |

#### CMNLI 中文自然语言推理 Chinese Multi-Genre NLI (Accuracy)：
| 模型 | matched | mismatched |  训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base	| 79.39 | 79.76 | batch=32, length=128, epoch=3 lr=2e-5 |
| BERT-wwm-ext-base	|81.41 |80.67|	batch=32, length=128, epoch=3 lr=2e-5 |
| ERNIE-base	|79.65 | 80.70 | batch=32, length=128, epoch=3 lr=2e-5 |
| ALBERT-xxlarge	|- | - | - |
| ALBERT-tiny	|72.71 | 72.72 | batch=32, length=128, epoch=3 lr=2e-5 |
| RoBERTa-large	| 82.11 | 81.73 | batch=16, length=128, epoch=3 lr=2e-5 |
| xlnet-mid	|78.15 |76.93 | batch=16, length=128, epoch=3 lr=2e-5 |
| RoBERTa-wwm-ext	|81.09 | 81.38 | batch=32, length=128, epoch=3 lr=2e-5  |
| RoBERTa-wwm-large-ext	|***83.4*** | ***83.42*** | batch=32, length=128, epoch=3 lr=2e-5  |

注：ALBERT-xlarge，在XNLI任务上训练暂时还存在有问题

#### COPA中文版  The Chinese Choice of Plausible Alternatives：

|         模型          | 开发集（dev %) | 测试集（test %) |                         训练参数                         |
| :-------------------: | :------------: | :-------------: | :------------------------------------------------------: |
|    ALBERT-xxlarge     |       -        |        -        |                            -                             |
|      ALBERT-tiny      |     52.00         |      49.80       |lr=1e-8, batch_size=8, max_seq_length=128, max_epochs=8 |
|       BERT-base       |     60.00      |      57.40       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |
|   BERT-wwm-ext-base   |     60.00      |      61.40       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |
|      ERNIE-base       |       60.00         |       65        |lr=2e-5, batch_size=32, max_seq_length=128, max_epochs=3|
|     RoBERTa-large     |     64.00      |      59.40       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |
|       XLNet-mid       |     56.00      |      53.80       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |
|    RoBERTa-wwm-ext    |     63.00      |      63.60       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |
| RoBERTa-wwm-large-ext |     66.00      |      59.40       | lr=1e-5, batch_size=12, max_seq_length=512, max_epochs=4 |

#### WSC Winograd模式挑战中文版  The Winograd Schema Challenge,Chinese Version：
| 模型 | 开发集（dev) | 测试集（test) | 训练参数 |
| :----:| :----: | :----: | :----: |
| ALBERT-xxlarge |    |    |  |
| ALBERT-tiny |  57.7(52.9)  |  58.5(52.1)  | lr=1e-4, batch_size=8, length=128, epoch=50   |
| BERT-base | 59.6（56.7)  | 62.0（57.9）  |  lr=2e-5, batch_size=8, length=128, epoch=50 |
| BERT-wwm-ext-base | 59.4(56.7) |  61.1(56.2) | lr=2e-5, batch_size=8, length=128, epoch=50   |
| ERNIE-base  | 58.1(54.9)| 60.8(55.9) | lr=2e-5, batch_size=8, length=128, epoch=50   |
| RoBERTa-large | 68.6(58.7)  | 72.7(63.6)  | lr=2e-5, batch_size=8, length=128, epoch=50   |
| XLNet-mid | 60.9(56.8）  |  64.4(57.3） | lr=2e-5, batch_size=8, length=128, epoch=50   |
| RoBERTa-wwm-ext | 67.2(57.7)  | 67.8(63.5)  | lr=2e-5, batch_size=8, length=128, epoch=50   |
| RoBERTa-wwm-large-ext |69.7(64.5) |  74.6(69.4) | lr=2e-5, batch_size=8, length=128, epoch=50   |

#### CSL 关键词识别  Keyword Recognition (Accuracy)：

|         模型          | 开发集（dev) | 测试集（test) |              训练参数              |
| :-------------------: | :----------: | :-----------: | :--------------------------------: |
|     ALBERT-xlarge     |    80.23     |     80.29     | batch_size=16, length=128, epoch=2 |
|     ALBERT-tiny       |    74.36     |     74.56     | batch_size=16, length=128, epoch=5 |
|       BERT-base       |    79.63     |     80.23     | batch_size=4, length=256, epoch=5  |
|   BERT-wwm-ext-base   |    80.60     |     81.00     | batch_size=4, length=256, epoch=5  |
|      ERNIE-base       |    79.43     |     79.10     | batch_size=4, length=256, epoch=5  |
|     RoBERTa-large     |    81.87     |     81.36     | batch_size=4, length=256, epoch=5  |
|       XLNet-mid       |    82.06     |     81.26     | batch_size=4, length=256, epoch=3  |
|    RoBERTa-wwm-ext    |    80.67     |     80.63     | batch_size=4, length=256, epoch=5  |
| RoBERTa-wwm-large-ext |    82.17     |     82.13     | batch_size=4, length=256, epoch=5  |

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

注: 现在榜上数据为cmrc2018完整测试集结果，之后CLUE将使用2k的测试集子集作为测试，而并非cmrc2018官方完整测试集。如需完整测试cmrc2018阅读理解数据集仍需通过cmrc2018平台提交(https://worksheets.codalab.org/worksheets/0x96f61ee5e9914aee8b54bd11e66ec647)。

#### CHID 成语阅读理解填空 Chinese IDiom Dataset for Cloze Test (Accuracy)：
| 模型 | 开发集（dev) | 测试集（test) |  训练参数 |
| :----:| :----: | :----: | :----: |
| BERT-base	|82.20 | 82.04 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| BERT-wwm-ext-base	|83.36 |82.9 |	batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| ERNIE-base	|82.46 | 82.28 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| ALBERT-xlarge	| 79.44 |79.55 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| ALBERT-tiny	| 43.47 |43.53 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| RoBERTa-large	| 85.31 |84.50 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| xlnet-mid	|83.76 | 83.47 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| RoBERTa-wwm-ext	|83.78 | 83.62 | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |
| RoBERTa-wwm-large-ext	|***85.81*** | ***85.37*** | batch=24, length=64, epoch=3, lr=2e-5, warmup=0.06 |


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

中文基准测评-成员 Members
---------------------------------------------------------------------
#### 顾问 Adviser：
张俊林，中国中文信息学会理事，中科院软件所博士，新浪微博机器学习团队AI Lab负责人。技术书籍《这就是搜索引擎：核心技术详解》（该书荣获全国第十二届优秀图书奖）、《大数据日知录：架构与算法》的作者。

崔一鸣，哈工大讯飞联合实验室（HFL）资深级研究员。中文机器阅读理解CMRC系列评测发起者，负责多个中文预训练模型项目，所领导的团队多次在SQuAD、CoQA、QuAC、HotpotQA等国际阅读理解评测中荣登榜首。

#### 会员 Charter Members（排名不分先后）：
徐亮，<a href='https://github.com/brightmart'>brightmart</a>，中文任务基准测评ChineseGLUE发起人。杭州实在智能算法专家，多个预训练模型中文版、文本分类开源项目作者。

Danny Lan，CMU博士、google AI 研究员，SOTA语言理解模型AlBERT第一作者。

徐国强，MIT博士，平安集团上海Gammalab负责人。

张轩玮，毕业于北京大学，目前在爱奇艺从事nlp有关的工作，之前做过热点聚合，文本分类，标签生成，机器翻译方面的工作。

谢炜坚，百度大数据部，算法工程师，从事NLP相关工作，包括任务驱动型对话、检索式问答、语义匹配、文本分类、情感分析等工作。

曹辰捷，平安金融壹账通，算法工程师，负责阅读理解和预训练相关业务，CMRC2019阅读理解冠军团队成员。

喻聪，来自杭州实在智能，主要研究多轮对话，意图识别，实体抽取，知识问答相关任务。

谢恩宁，大搜车，围绕汽车领域语对话机器人，负责NLU部分。

李露，华中师范大学研究生，曾参与筹备中文自然语言推理的数据集；暑期在平安科技实习，主要负责利用自然语言处理模型进行序列标注和情感分类任务。

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

4、<a href="https://arxiv.org/pdf/1809.05053.pdf">XNLI: Evaluating Cross-lingual Sentence Representations</a>

5、<a href="https://github.com/fate233/toutiao-text-classfication-dataset">TNES: toutiao-text-classfication-dataset</a>

6、<a href="https://github.com/brightmart/nlp_chinese_corpus">nlp_chinese_corpus: 大规模中文自然语言处理语料 Large Scale Chinese Corpus for NLP</a>

7、<a href="https://github.com/InsaneLife/ChineseNLPCorpus">ChineseNLPCorpus</a>

8、<a href="https://arxiv.org/abs/1909.11942">ALBERT: A Lite BERT For Self-Supervised Learning Of Language Representations</a>

9、<a href="https://arxiv.org/pdf/1810.04805.pdf">BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding</a>

10、<a href="https://arxiv.org/pdf/1907.11692.pdf">RoBERTa: A Robustly Optimized BERT Pretraining Approach</a>
