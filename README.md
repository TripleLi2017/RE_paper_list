# 关系抽取任务调研

## 任务定义

  关系抽取是信息抽取中的一项任务——给定文本中出现的实体对，识别它们的关系并分类到具体的类别。主要方法可以分为监督和远程监督两大类：

	监督：分类问题．需要预先了解语料库中所有关系的种类,并通过人工对数据进行标注,建立训练语料库。主要有流水线学习(pipeline)和联合学习(joint)两种。
	 
	远程监督：远程监督的基本假设是如果２个实体在己知知识库中存在着某种关系,那么涉及这２个实体的所有句子都会以某种方式表达这种关系。该方法无需事先人工定义实体关系的类型,可以自动地抽取大量的实体对，可以方便地移植到别的领域,适合针对大规模地网络文本数据进行实体间的关系抽取。

### 任务示例

输入:

```
[李晓华]和她的丈夫[王大牛]前日一起去[英国]旅行了。
```

输出:

```
(entity1: 李晓华, entity2: 王大牛, relation: 夫妻) 
(entity1: 李晓华, entity2: 英国, relation: 位于)
(entity1: 王大牛, entity2: 英国, relation: 位于
```

## 评估指标

精确率 (Precision), 召回率 (Recall), F1。准确率是对于给定的测试数据集,分类器正确分类为正类的样本数与全部正类样本数之比;召回率则是对于给定的测试数据集,预测正确的正类与所有正类数据的比值;而F１值则是准确率和召回率的调和平均值,可以对系统的性能进行综合性的评价．

$$
Precision＝\frac{TP}{TP＋FP}
$$

$$
Recall＝\frac{TP}{TP＋FN}
$$

$$
F１＝\frac{２×Precision×Recall}{Precision＋Recall}
$$

１) TP(true positive)．原本是正类,预测结果为正类(正确预测为正类)．

２) FP(false positive)．原本是负类,预测结果为正类(错误预测为正类)．

３) TN(true negative)．原本是负类,预测结果为负类(正确预测为负类)．

４) FN(false negative)．原本是正类,预测结果为负类(错误预测为负类)．



* 针对开放领域的关系抽取,目前还缺少公认的评测体系,一般通过考查抽取关系的准确性以及综
  合考虑算法的时间复杂度、空间复杂度等因素来评价关系抽取模型的性能。

## 数据调研

### ACE

**数据来源：**ACE2005语料库是语言数据联盟(LDC)发布的由实体，关系和事件注释组成的各种类型的数据，包括英语，阿拉伯语和中文培训数据，目标是开发自动内容提取技术，支持以文本形式自动处理人类语言。ACE测评会议提出实体关系是实体之间显式或者隐式的语义联系,因此需要预先定义实体关系的类型,然后识别实体之间是否存在语义关系,进而判定属于哪一种预定义的关系类型。

ACE提供每个实体的类型，主要分为PER，ORG，GPE，LOC，VEH，FAC，WEA7类。

ACE语料库的获取链接：https://catalog.ldc.upenn.edu/LDC2006T06

#### 数据来源：

![ace_data](/Users/lilili/Desktop/毕设/paper_pic/ace_data.png)

#### 数据格式：

* Source Text (.sgm) Files
   * 这些文件是SGM格式的源文本文件,.sgm文件是UTF-8编码的
   * AG (.ag.xml) Files
     * 这些是使用LDC的注释工具创建的注释文件，这些文件被转换为对应的.apf.xml文件。
       ID table (.tab) Files
          - 这些文件通过使用ag.xml文件和相应的apf.xml文件存储ID们之间的映射表
            ACE Program Format (APF) (.apf.xml) Files
  * 主要使用的注释文件，xml格式。

#### 数据集关系类型及比例：

![ACE](./pic/ace-data.png)

##### SOTA：

|      | Model                    | RE Micro F1 | Paper                                                        | Code                                                         | Year |
| :--- | ------------------------ | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 1    | PL-Marker                | 70.5        | [Pack Together: Entity and Relation  Extraction with Levitated Marker](https://paperswithcode.com/paper/pack-together-entity-and-relation-extraction) | [code](https://github.com/thunlp/pl-marker)                  | 2021 |
| 2    | Ours: cross-sentence ALB | 67.0        | [A Frustratingly Easy Approach for Entity  and Relation Extraction](https://paperswithcode.com/paper/a-frustratingly-easy-approach-for-joint) | [code](https://github.com/princeton-nlp/PURE)                | 2020 |
| 3    | PFN                      | 66.8        | [HySPA: Hybrid Span Generation for Scalable Text-to-Graph Extraction](https://paperswithcode.com/paper/hyspa-hybrid-span-generation-for-scalable) | [code](https://github.com/Coopercoppers/PFN)                 | 2021 |
| 4    | Table-Sequence           | 64.3        | [Two are Better than One: Joint Entity and  Relation Extraction with Table-Sequence Encoders](https://paperswithcode.com/paper/two-are-better-than-one-joint-entity-and) | [code](https://github.com/LorrinWWW/two-are-better-than-one) | 2020 |
| 5    | TriMF                    | 62.77       | [A Trigger-Sense Memory Flow Framework for  Joint Entity and Relation Extraction](https://paperswithcode.com/paper/a-trigger-sense-memory-flow-framework-for) | [code](https://github.com/tricktreat/trimf)                  | 2021 |
| 6    | MRC4ERE++                | 62.1        | [Asking Effective and Diverse Questions: A  Machine Reading Comprehension based Framework for Joint Entity-Relation  Extraction](https://paperswithcode.com/paper/asking-effective-and-diverse-questions-a) | [code](https://github.com/TanyaZhao/MRC4ERE)                 | 2020 |
| 7    | Multi-turn QA            | 60.2        | [Entity-Relation Extraction as Multi-Turn  Question Answering](https://paperswithcode.com/paper/entity-relation-extraction-as-multi-turn) | [code](https://github.com/ShannonAI/Entity-Relation-As-Multi-Turn-QA) | 2019 |
| 8    | MRT                      | 59.6        | [Extracting Entities and Relations with  Joint Minimum Risk Training](https://paperswithcode.com/paper/extracting-entities-and-relations-with-joint) |                                                              | 2018 |
| 9    | GCN                      | 59.1        | [Joint Type Inference on Entities and  Relations via Graph Convolutional Networks](https://paperswithcode.com/paper/joint-type-inference-on-entities-and) |                                                              | 2019 |
| 10   | Global                   | 57.5        | [End-to-End Neural Relation Extraction  with Global Optimization](https://paperswithcode.com/paper/end-to-end-neural-relation-extraction-with) |                                                              | 2017 |
| 11   | SPTree                   | 55.6        | [End-to-End Relation Extraction using  LSTMs on Sequences and Tree Structures](https://paperswithcode.com/paper/end-to-end-relation-extraction-using-lstms-on) | [code](https://github.com/tticoin/LSTM-ER)                   | 2016 |
| 12   | Attention                | 53.6        | [Going out on a limb: Joint Extraction of  Entity Mentions and Relations without Dependency Trees](https://paperswithcode.com/paper/going-out-on-a-limb-joint-extraction-of) |                                                              | 2017 |
| 13   | Joint w/ Global          | 49.5        | [Incremental Joint Extraction of Entity  Mentions and Relations](https://paperswithcode.com/paper/incremental-joint-extraction-of-entity) |                                                              | 2014 |

### semeval 

SemEval数据集来自于2010年的国际语义评测大会中 Task 8:” Multi-Way Classification of Semantic Relations Between Pairs of Nominals “。SemEval会议定义了最初９种常见名词及其关系(原因影响、仪器机构、产品生产者、含量包含者、实体来源地、实体目的地、部分整体、成员集合、行为主题) + 1类其他。

##### 数据链接：https://github.com/thunlp/OpenNRE/tree/master/benchmark

文档描述：[官方文档](https://docs.google.com/document/d/1QO_CnmvNRnYwNWu1-QCAeR5ToQYkXUqFeAJbdEhsq7w/preview)

##### 数据分布：

![semeval](./pic/semeval.png)

SOTA：

|      | Model                    | F1    | Paper                                                        | Code                                                         | Year |
| ---- | ------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 1    | QA                       | 91.9  | [Relation Classification as Two-way  Span-Prediction](https://paperswithcode.com/paper/relation-extraction-as-two-way-span) |                                                              | 2020 |
| 2    | RIFRE                    | 91.3  | [Representation Iterative Fusion based on  Heterogeneous Graph Neural Network for Joint Entity and Relation Extraction](https://paperswithcode.com/paper/representation-iterative-fusion-based-on) | [code](https://github.com/zhao9797/RIFRE)                    | 2021 |
| 3    | REDN                     | 91    | [Downstream Model Design of Pre-trained  Language Model for Relation Extraction Task](https://paperswithcode.com/paper/downstream-model-design-of-pre-trained) | [code](https://github.com/slczgwh/REDN)                      | 2020 |
| 4    | Skeleton-Aware BERT      | 90.36 | [Enhancing Relation Extraction Using  Syntactic Indicators and Sentential Contexts](https://paperswithcode.com/paper/enhancing-relation-extraction-using-syntactic) | [code](https://github.com/taoqiongxing/Enhancing-Relation-Extraction-using-Syntactic-Indicators-and-Sentential-Contexts) | 2019 |
| 5    | KnowPrompt               | 90.3  | [KnowPrompt: Knowledge-aware Prompt-tuning  with Synergistic Optimization for Relation Extraction](https://paperswithcode.com/paper/adaprompt-adaptive-prompt-based-finetuning) | [code](https://github.com/zjunlp/KnowPrompt)                 | 2021 |
| 6    | EPGNN                    | 90.2  | [Improving Relation Classification by  Entity Pair Graph](https://paperswithcode.com/paper/improving-relation-classification-by-entity) |                                                              | 2019 |
| 7    | BERTEM+MTB               | 89.5  | [Matching the Blanks: Distributional  Similarity for Relation Learning](https://paperswithcode.com/paper/matching-the-blanks-distributional-similarity) | [code](https://github.com/plkmo/BERT-Relation-Extraction)    | 2019 |
| 8    | R-BERT                   | 89.25 | [Enriching Pre-trained Language Model with  Entity Information for Relation Classification](https://paperswithcode.com/paper/enriching-pre-trained-language-model-with) | [code](https://paperswithcode.com/paper/enriching-pre-trained-language-model-with#code) | 2020 |
| 9    | KnowBert-W+W             | 89.1  | [Knowledge Enhanced Contextual Word  Representations](https://paperswithcode.com/paper/knowledge-enhanced-contextual-word) | [code](https://paperswithcode.com/paper/knowledge-enhanced-contextual-word#code) | 2019 |
| 10   | Entity-Aware BERT        | 89    | [Extracting Multiple-Relations in One-Pass  with Pre-Trained Transformers](https://paperswithcode.com/paper/extracting-multiple-relations-in-one-pass) | [code](https://github.com/helloeve/mre-in-one-pass)          | 2019 |
| 11   | Att-Pooling-CNN          | 88    | [Relation Classification via Multi-Level  Attention CNNs](https://paperswithcode.com/paper/relation-classification-via-multi-level) |                                                              | 2016 |
| 12   | SpanRel                  | 87.4  | [Generalizing Natural Language Analysis  through Span-relation Representations](https://paperswithcode.com/paper/generalizing-natural-language-analysis-1) | [code](https://github.com/jzbjyb/SpanRel)                    | 2019 |
| 13   | TRE                      | 87.1  | [Improving Relation Extraction by  Pre-trained Language Representations](https://paperswithcode.com/paper/improving-relation-extraction-by-pre-trained-1) | [code](https://github.com/DFKI-NLP/TRE)                      | 2019 |
| 14   | Entity Attention Bi-LSTM | 85.2  | [Semantic Relation Classification via  Bidirectional LSTM Networks with Entity-aware Attention using Latent Entity  Typing](https://paperswithcode.com/paper/semantic-relation-classification-via-1) | [code](https://github.com/roomylee/entity-aware-relation-classification) | 2019 |
| 15   | Attention CNN            | 84.3  | [Attention-Based Convolutional Neural  Network for Semantic Relation Extraction](https://paperswithcode.com/paper/attention-based-convolutional-neural-network-2) | [code](https://github.com/onehaitao/Attention-CNN-relation-extraction) | 2016 |
| 16   | CR-CNN                   | 84.1  | [Classifying Relations by Ranking with  Convolutional Neural Networks](https://paperswithcode.com/paper/classifying-relations-by-ranking-with) | [code](https://github.com/onehaitao/CR-CNN-relation-extraction) | 2015 |
| 17   | Attention Bi-LSTM        | 84    | [Attention-Based Bidirectional Long  Short-Term Memory Networks for Relation Classification](https://paperswithcode.com/paper/attention-based-bidirectional-long-short-term) | [code](https://github.com/onehaitao/Att-BLSTM-relation-extraction) | 2016 |
| 18   | CNN                      | 82.7  | [Relation Classification via Convolutional  Deep Neural Network](https://paperswithcode.com/paper/relation-classification-via-convolutional) | [Code](https://github.com/onehaitao/CNN-relation-extraction) | 2014 |
| 19   | Bi-LSTM                  | 82.7  | [Bidirectional Long Short-Term Memory  Networks for Relation Classification](https://paperswithcode.com/paper/bidirectional-long-short-term-memory-networks) |                                                              | 2015 |

## 远程监督数据集

### NYT

**数据来源：**NYT10是在基于远程监督的关系抽取任务上最常用的数据集，NYT10数据集来自于10年的论文Modeling Relations and Their Mentions withoutLabeled Text，是由NYT corpus 同Freebase远程监督得到, 样本的是根据包的形式分布的及含有相同实体的数据集分布在一起.

![NYT1](./pic/NYT1.png)

数据集中一共包含52+1（包括NA）个关系，各个关系在样本中的分布如下：

| relations                                              | size_of_train | size_of_test |
| ------------------------------------------------------ | ------------- | ------------ |
| /location/fr_region/capital                            | 1             | 0            |
| /location/cn_province/capital                          | 2             | 0            |
| /location/in_state/administrative_capital              | 4             | 0            |
| /base/locations/countries/states_provinces_within      | 0             | 1            |
| /business/company/founders                             | 901           | 95           |
| /people/person/place_of_birth                          | 4053          | 162          |
| /people/deceased_person/place_of_death                 | 2422          | 68           |
| /location/it_region/capital                            | 22            | 0            |
| /people/family/members                                 | 4             | 0            |
| /people/profession/people_with_this_profession         | 2             | 0            |
| /location/neighborhood/neighborhood_of                 | 9275          | 68           |
| /location/in_state/legislative_capital                 | 4             | 0            |
| /sports/sports_team/location                           | 294           | 10           |
| /people/person/religion                                | 202           | 6            |
| /location/in_state/judicial_capital                    | 3             | 0            |
| /business/company_advisor/companies_advised            | 2             | 8            |
| /people/family/country                                 | 6             | 0            |
| /time/event/locations                                  | 4             | 4            |
| /business/company/place_founded                        | 648           | 20           |
| /location/administrative_division/country              | 7286          | 424          |
| /people/ethnicity/included_in_group                    | 7             | 0            |
| /location/br_state/capital                             | 4             | 2            |
| /location/mx_state/capital                             | 1             | 0            |
| /location/province/capital                             | 39            | 11           |
| /people/person/nationality                             | 9733          | 723          |
| /business/person/company                               | 7336          | 302          |
| /business/shopping_center_owner/shopping_centers_owned | 1             | 0            |
| /business/company/advisors                             | 9             | 8            |
| /business/shopping_center/owner                        | 1             | 0            |
| /location/country/languages_spoken                     | 0             | 3            |
| /people/deceased_person/place_of_burial                | 24            | 9            |
| /location/us_county/county_seat                        | 110           | 23           |
| /people/ethnicity/geographic_distribution              | 86            | 136          |
| /people/person/place_lived                             | 8907          | 450          |
| /business/company/major_shareholders                   | 328           | 46           |
| /broadcast/producer/location                           | 71            | 0            |
| /location/us_state/capital                             | 798           | 39           |
| /broadcast/content/location                            | 8             | 0            |
| /business/business_location/parent_company             | 19            | 0            |
| /location/jp_prefecture/capital                        | 2             | 0            |
| /film/film/featured_film_locations                     | 18            | 2            |
| /people/place_of_interment/interred_here               | 24            | 9            |
| /location/de_state/capital                             | 7             | 0            |
| /people/person/profession                              | 10            | 0            |
| /business/company/locations                            | 19            | 0            |
| /location/country/capital                              | 8883          | 553          |
| /location/location/contains                            | 66721         | 2793         |
| /people/person/ethnicity                               | 148           | 13           |
| /location/country/administrative_divisions             | 7286          | 424          |
| /people/person/children                                | 622           | 30           |
| /film/film_location/featured_in_films                  | 18            | 2            |
| /film/film_festival/location                           | 4             | 0            |
| NA                                                     | 385664        | 166004       |
| 合计                                                   | 522043        | 172448       |

SOTA:

|      | Model               | F1   | Paper                                                        | Code                                                         | Years |
| ---- | ------------------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----- |
| 1    | REBEL               | 93.4 | [REBEL:   Relation Extraction By End-to-end Language generation](https://paperswithcode.com/paper/rebel-relation-extraction-by-end-to-end) | [code](https://paperswithcode.com/paper/rebel-relation-extraction-by-end-to-end#code) | 2021  |
| 2    | REBEL               | 93.1 | [REBEL:   Relation Extraction By End-to-end Language generation](https://paperswithcode.com/paper/rebel-relation-extraction-by-end-to-end) | [code](https://paperswithcode.com/paper/rebel-relation-extraction-by-end-to-end#code) | 2021  |
| 3    | SPN                 | 92.5 | [Joint   Entity and Relation Extraction with Set Prediction Networks](https://paperswithcode.com/paper/joint-entity-and-relation-extraction-with-set) | [code](https://paperswithcode.com/paper/joint-entity-and-relation-extraction-with-set#code) | 2020  |
| 4    | TDEER               | 92.5 | [TDEER:   An Efficient Translating Decoding Schema for Joint Extraction of Entities and   Relations](https://paperswithcode.com/paper/tdeer-an-efficient-translating-decoding) | [code](https://paperswithcode.com/paper/tdeer-an-efficient-translating-decoding#code) | 2021  |
| 5    | PFN                 | 92.4 | [A   Partition Filter Network for Joint Entity and Relation Extraction](https://paperswithcode.com/paper/a-partition-filter-network-for-joint-entity) | [code](https://paperswithcode.com/paper/representation-iterative-fusion-based-on#code) | 2021  |
| 6    | RIFRE               | 92   | [Representation   Iterative Fusion based on Heterogeneous Graph Neural Network for Joint Entity   and Relation Extraction](https://paperswithcode.com/paper/representation-iterative-fusion-based-on) | [code](https://paperswithcode.com/paper/representation-iterative-fusion-based-on#code) | 2021  |
| 7    | TPLinker            | 91.9 | [TPLinker:   Single-stage Joint Extraction of Entities and Relations Through Token Pair   Linking](https://paperswithcode.com/paper/tplinker-single-stage-joint-extraction-of) | [code](https://paperswithcode.com/paper/tplinker-single-stage-joint-extraction-of#code) | 2020  |
| 8    | HBT                 | 89.6 | [A   Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://paperswithcode.com/paper/a-novel-hierarchical-binary-tagging-framework) | [code](https://paperswithcode.com/paper/a-novel-hierarchical-binary-tagging-framework#code) | 2019  |
| 9    | CGT                 | 89.1 | [Contrastive   Triple Extraction with Generative Transformer](https://paperswithcode.com/paper/contrastive-triple-extraction-with-generative) |                                                              | 2020  |
| 10   | BiTT                | 88.9 | [BiTT:   Bidirectional Tree Tagging for Joint Extraction of Overlapping Entities and   Relations](https://paperswithcode.com/paper/a-bidirectional-tree-tagging-scheme-for) |                                                              | 2020  |
| 11   | RIN                 | 87.8 | [Recurrent   Interaction Network for Jointly Extracting Entities and Classifying Relations](https://paperswithcode.com/paper/recurrent-interaction-network-for-jointly) |                                                              | 2020  |
| 12   | RSAN                | 84.6 | [A   Relation-Specific Attention Network for Joint Entity and Relation Extraction](https://paperswithcode.com/paper/a-relation-specific-attention-network-for) | [code](https://paperswithcode.com/paper/a-relation-specific-attention-network-for#code) | 2020  |
| 13   | ETL-Span            | 78   | [Joint   Extraction of Entities and Relations Based on a Novel Decomposition Strategy](https://paperswithcode.com/paper/joint-extraction-of-entities-and-relations-3) | [code](https://paperswithcode.com/paper/joint-extraction-of-entities-and-relations-3#code) | 2019  |
| 14   | CopyRE' OneDecoder  | 72.2 | [CopyMTL:   Copy Mechanism for Joint Extraction of Entities and Relations with Multi-Task   Learning](https://paperswithcode.com/paper/copymtl-copy-mechanism-for-joint-extraction) | [code](https://paperswithcode.com/paper/copymtl-copy-mechanism-for-joint-extraction#code) | 2019  |
| 15   | CopyRE MultiDecoder | 58.7 | [Extracting   Relational Facts by an End-to-End Neural Model with Copy Mechanism](https://paperswithcode.com/paper/extracting-relational-facts-by-an-end-to-end) | [code](https://paperswithcode.com/paper/extracting-relational-facts-by-an-end-to-end#code) | 2018  |
| 16   | PA                  | 53.8 | [Joint   extraction of entities and overlapping relations using position-attentive   sequence labeling](https://paperswithcode.com/paper/joint-extraction-of-entities-and-overlapping) |                                                              | 2019  |
| 17   | NovelTagging        | 42   | [Joint   Extraction of Entities and Relations Based on a Novel Tagging Scheme](https://paperswithcode.com/paper/joint-extraction-of-entities-and-relations) | [code](https://paperswithcode.com/paper/joint-extraction-of-entities-and-relations#code) | 2017  |

### tacred

**数据来源：**TACRED由Stanford NLP组开发，其中包含在2009 - 2014年期间NIST TAC KBP English slot filling evaluations中所有的英文新闻以及web文本。

##### 数据分布：

| **关系**                                 | **数量** | **比例**  | **train** | **dev** | **test** |
| ---------------------------------------- | -------- | --------- | --------- | ------- | -------- |
| no_relation                              | 84491    | 79.51%    | 55112     | 17195   | 12184    |
| org:alternate_names                      | 1359     | 1.28%     | 808       | 338     | 213      |
| org:city_of_headquarters_of_headquarters | 573      | 0.54%     | 382       | 109     | 82       |
| org:country_of_headquarters              | 753      | 0.71%     | 468       | 177     | 108      |
| org:dissolved                            | 33       | 0.03%     | 23        | 8       | 2        |
| org:founded                              | 166      | 0.16%     | 91        | 38      | 37       |
| org:founded_by                           | 268      | 0.25%     | 124       | 76      | 68       |
| org:member_of                            | 171      | 0.16%     | 122       | 31      | 18       |
| org:members                              | 286      | 0.27%     | 170       | 85      | 31       |
| org:number_of_employees/members          | 121      | 0.11%     | 75        | 27      | 19       |
| org:parents                              | 444      | 0.42%     | 286       | 96      | 62       |
| org:political/religious_affiliation      | 125      | 0.12%     | 105       | 10      | 10       |
| org:shareholders                         | 144      | 0.14%     | 76        | 55      | 13       |
| org:stateorprovince_of_headquarters      | 350      | 0.33%     | 229       | 70      | 51       |
| org:subsidiaries                         | 453      | 0.43%     | 296       | 113     | 44       |
| org:top_members/employees                | 2770     | 2.61%     | 1890      | 534     | 346      |
| org:website                              | 223      | 0.21%     | 111       | 86      | 26       |
| per:age                                  | 833      | 0.78%     | 390       | 243     | 200      |
| per:alternate_names                      | 153      | 0.14%     | 104       | 38      | 11       |
| per:cause_of_death                       | 337      | 0.32%     | 117       | 168     | 52       |
| per:charges                              | 280      | 0.26%     | 72        | 105     | 103      |
| per:children                             | 347      | 0.33%     | 211       | 99      | 37       |
| per:cities_of_residence                  | 742      | 0.70%     | 374       | 179     | 189      |
| per:city_of_birth                        | 103      | 0.10%     | 65        | 33      | 5        |
| per:city_of_death                        | 227      | 0.21%     | 81        | 118     | 28       |
| per:countries_of_residence               | 819      | 0.77%     | 445       | 226     | 148      |
| per:country_of_birth                     | 53       | 0.05%     | 28        | 20      | 5        |
| per:country_of_death                     | 61       | 0.06%     | 6         | 46      | 9        |
| per:date_of_birth                        | 103      | 0.10%     | 63        | 31      | 9        |
| per:date_of_death                        | 394      | 0.37%     | 134       | 206     | 54       |
| per:employee_of                          | 2163     | 2.04%     | 1524      | 375     | 264      |
| per:origin                               | 667      | 0.63%     | 325       | 210     | 132      |
| per:other_family                         | 319      | 0.30%     | 179       | 80      | 60       |
| per:parents                              | 296      | 0.28%     | 152       | 56      | 88       |
| per:religion                             | 153      | 0.14%     | 53        | 53      | 47       |
| per:schools_attended                     | 229      | 0.22%     | 149       | 50      | 30       |
| per:siblings                             | 250      | 0.24%     | 165       | 30      | 55       |
| per:spouse                               | 483      | 0.45%     | 258       | 159     | 66       |
| 00 0per:stateorprovince_of_birth         | 72       | 0.07%     | 38        | 26      | 8        |
| per:stateorprovince_of_death             | 104      | 0.10%     | 49        | 41      | 14       |
| per:stateorprovinces_of_residence        | 484      | 0.46%     | 331       | 72      | 81       |
| per:title                                | 3862     | 3.63%     | 2443      | 919     | 500      |
| Total                                    | 106264   | 10000.00% | 68124     | 22631   | 15509    |

SOTA：

|      | model              | f1    | paper                                                        | Code                                                         | Year |
| ---- | ------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| 1    | RECENT+SpanBERT    | 75.2  | [Relation Classification with Entity Type  Restriction](https://arxiv.org/pdf/2105.08393v1.pdf) |                                                              | 2021 |
| 2    | KU_NLP             | 75    | [Improving Sentence-Level Relation  Extraction through Curriculum Learning](https://arxiv.org/pdf/2107.09332v2.pdf) |                                                              | 2021 |
| 3    | Relation Reduction | 74.8  | [Relation Classification as Two-way  Span-Prediction](https://arxiv.org/pdf/2107.09332v2.pdf) |                                                              | 2020 |
| 5    | NLI_DeBERTa        | 73.9  | [Label Verbalization and Entailment for  Effective Zero- and Few-Shot Relation Extraction](https://paperswithcode.com/paper/label-verbalization-and-entailment-for) | [code](https://paperswithcode.com/paper/label-verbalization-and-entailment-for#code) | 2021 |
| 6    | LUKE               | 72.7  | [LUKE: Deep Contextualized Entity  Representations with Entity-aware Self-attention](https://paperswithcode.com/paper/luke-deep-contextualized-entity) | [code](https://paperswithcode.com/paper/luke-deep-contextualized-entity#code) | 2020 |
| 7    | DeNERT-KG          | 72.4  | [DeNERT-KG: Named Entity and Relation  Extraction Model Using DQN, Knowledge Graph, and BERT](https://paperswithcode.com/paper/denert-kg-named-entity-and-relation) |                                                              | 2020 |
| 8    | K-ADAPTER          | 72.04 | [K-Adapter: Infusing Knowledge into  Pre-Trained Models with Adapters](https://paperswithcode.com/paper/k-adapter-infusing-knowledge-into-pre-trained) | [code](https://paperswithcode.com/paper/k-adapter-infusing-knowledge-into-pre-trained#code) | 2020 |
| 9    | KEPLER             | 71.7  | [KEPLER: A Unified Model for Knowledge  Embedding and Pre-trained Language Representation](https://arxiv.org/pdf/1911.06136v3.pdf) | [code](https://paperswithcode.com/paper/kepler-a-unified-model-for-knowledge#code) | 2019 |
| 10   | BERTEM+MTB         | 71.5  | [Matching the Blanks: Distributional  Similarity for Relation Learning](https://arxiv.org/pdf/1906.03158v1.pdf) | [code](https://paperswithcode.com/paper/matching-the-blanks-distributional-similarity#code) | 2019 |
| 11   | KnowBert-W+W       | 71.5  | [Knowledge Enhanced Contextual Word  Representations](https://arxiv.org/pdf/1909.04164v2.pdf) | [code](https://paperswithcode.com/paper/knowledge-enhanced-contextual-word#code) | 2019 |
| 12   | DG-SpanBERT-large  | 71.5  | [Efficient long-distance relation  extraction with DG-SpanBERT](https://arxiv.org/pdf/2004.03636v1.pdf) |                                                              | 2020 |
| 13   | NLI_RoBERTa        | 71    | [Label Verbalization and Entailment for  Effective Zero- and Few-Shot Relation Extraction](https://arxiv.org/pdf/2109.03659v1.pdf) | [Code](https://github.com/osainz59/Ask2Transformers)         | 2021 |
| 15   | GDPNet             | 70.5  | [GDPNet: Refining Latent Multi-View Graph  for Relation Extraction](https://arxiv.org/pdf/2012.06780v1.pdf) | [code](https://github.com/XueFuzhao/GDPNet)                  | 2020 |
| 18   | C-GCN + PA-LSTM    | 68.2  | [Graph Convolution over Pruned Dependency  Trees Improves Relation Extraction](https://arxiv.org/pdf/1809.10185v1.pdf) | [code](https://github.com/qipeng/gcn-over-pruned-trees)      | 2018 |
| 19   | C-AGGCN            | 68.2  | [Attention Guided Graph Convolutional  Networks for Relation Extraction](https://arxiv.org/pdf/1906.07510v8.pdf) | [code](https://github.com/Cartus/AGGCN_TACRED)               | 2019 |
| 22   | SA-LSTM+D          | 67.6  | Beyond Word Attention: Using Segment  Attention in Neural Relation Extraction |                                                              | 2019 |
| 25   | GCN + PA-LSTM      | 67.1  | [Graph Convolution over Pruned Dependency  Trees Improves Relation Extraction](https://arxiv.org/pdf/1809.10185v1.pdf) | [code](https://github.com/qipeng/gcn-over-pruned-trees)      | 2018 |
| 26   | C-SGC              | 67    | [Simplifying Graph Convolutional Networks](https://arxiv.org/pdf/1902.07153v2.pdf) | [code](https://github.com/Tiiiger/SGC)                       | 2019 |
| 27   | C-GCN              | 66.4  | [Graph Convolution over Pruned Dependency  Trees Improves Relation Extraction](https://arxiv.org/pdf/1809.10185v1.pdf) | [code](https://github.com/qipeng/gcn-over-pruned-trees)      | 2018 |
| 28   | PA-LSTM            | 65.1  | [Position-aware Attention and Supervised  Data Improve Slot Filling](https://aclanthology.org/D17-1004.pdf) | [code](https://github.com/yuhaozhang/tacred-relation)        | 2017 |
| 29   | AGGCN              | 65.1  | [Attention Guided Graph Convolutional  Networks for Relation Extraction](https://arxiv.org/pdf/1906.07510v8.pdf) | [code](https://github.com/Cartus/AGGCN_TACRED)               | 2019 |
| 30   | GCN                | 64    | [Graph Convolution over Pruned Dependency  Trees Improves Relation Extraction](https://arxiv.org/pdf/1809.10185v1.pdf) | [code](https://github.com/qipeng/gcn-over-pruned-trees)      | 2018 |



### [DocRED](https://github.com/thunlp/DocRED)

**数据来源：**DocRED（Document-Level Relation Extraction Dataset）由维基百科和维基数据构建，每个文档都对命名实体提及、共指信息、句内和句间关系进行了人工注释。 DocRED 需要读取文档中的多个句子来提取实体，并通过综合文档的所有信息来推断它们的关系。除了人工标注的数据外，该数据集还提供了大规模的远程监督数据。

DocRED 包含5053个维基百科文档， 132,375 个实体和 96个关系。除了人工标注的数据外，该数据集还提供了超过 101,873 个文档的大规模远程监督数据。

##### SOTA：

| Rank | Model                              | F1    | Paper                                                        | Code                                               | Year |
| ---- | ---------------------------------- | ----- | ------------------------------------------------------------ | -------------------------------------------------- | ---- |
| 1    | SSAN-RoBERTa-large+Adaptation      | 65.92 | [Entity Structure Within and Throughout:  Modeling Mention Dependencies for Document-Level Relation Extraction](https://arxiv.org/pdf/2102.10249v1.pdf) | [code](https://github.com/PaddlePaddle/Research)   | 2021 |
| 2    | SAISBAll-RoBERTa-large             | 65.11 | [SAIS: Supervising and Augmenting  Intermediate Steps for Document-Level Relation Extraction](https://arxiv.org/pdf/2109.12093v1.pdf) |                                                    | 2021 |
| 3    | Eider-RoBERTa-large                | 64.79 | [Eider: Evidence-enhanced Document-level  Relation Extraction](https://arxiv.org/pdf/2106.08657v1.pdf) |                                                    | 2021 |
| 4    | DocuNet-RoBERTa-large              | 64.55 | [Document-level Relation Extraction as  Semantic Segmentation](https://arxiv.org/pdf/2106.03618v2.pdf) | [code](https://github.com/zjunlp/DocuNet)          | 2021 |
| 5    | ATLOP-RoBERTa-large                | 63.4  | [Document-Level Relation Extraction with  Adaptive Thresholding and Localized Context Pooling](https://arxiv.org/pdf/2010.11304v3.pdf) | [code](https://github.com/wzhouad/ATLOP)           | 2020 |
| 7    | GAIN-BERT-large                    | 62.76 | [Double Graph Based Reasoning for  Document-level Relation Extraction](https://github.com/DreamInvoker/GAIN) | [code](https://github.com/DreamInvoker/GAIN)       | 2020 |
| 13   | ATLOP-BERT-base                    | 61.3  | [Document-Level Relation Extraction with  Adaptive Thresholding anåd Localized Context Pooling](https://arxiv.org/pdf/2010.11304v3.pdf) | [code](https://arxiv.org/pdf/2010.11304v3.pdf)     | 2020 |
| 14   | GAIN-BERT-base                     | 61.24 | Double Graph Based Reasoning for  Document-level Relation Extraction |                                                    | 2020 |
| 15   | JEREX-BERT-base                    | 60.4  | An End-to-end Model for Entity-level  Relation Extraction using Multi-instance Learning |                                                    | 2021 |
| 16   | CorefRoBERTa-large                 | 60.25 | [Coreferential Reasoning Learning for  Language Representation](https://arxiv.org/pdf/2004.06870v2.pdf) | [code](https://github.com/thunlp/KernelGAT)        | 2020 |
| 17   | SSAN-RoBERTa-base                  | 59.94 | [Entity Structure Within and Throughout:  Modeling Mention Dependencies for Document-Level Relation Extraction](https://arxiv.org/pdf/2102.10249v1.pdf) | [**code](https://github.com/PaddlePaddle/Research) | 2021 |
| 18   | CFER-BERT-base                     | 59.82 | Coarse-to-Fine Entity Representations for  Document-level Relation Extraction |                                                    | 2020 |
| 19   | HeterGSAN+Reconstruction+BERT-base | 59.45 | Document-Level Relation Extraction with  Reconstruction      |                                                    | 2020 |
| 20   | LSR+BERT-base                      | 59.05 | Reasoning with Latent Structure  Refinement for Document-Level Relation Extraction |                                                    | 2020 |
| 21   | GLRE-XLNet-Large                   | 59    | Global-to-Local Neural Networks for  Document-Level Relation Extraction |                                                    | 2020 |
| 22   | CorefBERT-large                    | 58.83 | Coreferential Reasoning Learning for  Language Representation |                                                    | 2020 |
| 23   | E2GRE-BERT-base                    | 58.72 | Entity and Evidence Guided Relation  Extraction for DocRED   |                                                    | 2020 |
| 24   | EncAttAgg                          | 58.7  | Improving Document-level Relation  Extraction via Contextualizing Mention Representations and Weighting Mention  Pairs |                                                    | 2020 |
| 25   | SSAN-BERT-base                     | 58.16 | Entity Structure Within and Throughout:  Modeling Mention Dependencies for Document-Level Relation Extraction |                                                    | 2021 |
| 26   | DUAL+BERT-base                     | 57.74 | Dual Supervision Framework for Relation  Extraction with Distant Supervision and Human Annotation |                                                    | 2020 |
| 27   | CorefBERT-base                     | 56.96 | [Coreferential Reasoning Learning for  Language Representation](https://arxiv.org/pdf/2004.06870v2.pdf) | [code](https://github.com/thunlp/KernelGAT)        | 2020 |
| 28   | DRN-GloVe                          | 56.33 | [Discriminative Reasoning for  Document-level Relation Extraction](https://arxiv.org/pdf/2106.01562v1.pdf) | [Code](https://github.com/xwjim/DRN)               | 2021 |
| 29   | Paths+BiLSTM-GloVe                 | 56.23 | Three Sentences Are All You Need: Local  Path Enhanced Document Relation Extraction |                                                    | 2021 |
| 30   | SIRE-GloVe                         | 55.96 | SIRE: Separate Intra- and  Inter-sentential Reasoning for Document-level Relation Extraction |                                                    | 2021 |
| 31   | CFER-GloVe                         | 55.75 | Coarse-to-Fine Entity Representations for  Document-level Relation Extraction |                                                    | 2020 |
| 32   | HIN-BERT-base                      | 55.6  | HIN: Hierarchical Inference Network for  Document-Level Relation Extraction |                                                    | 2020 |
| 33   | HeterGSAN+Reconstruction           | 55.23 | [Document-Level Relation Extraction with  Reconstruction](https://arxiv.org/pdf/2012.11384v1.pdf) | [code](https://github.com/xwjim/DocRE-Rec)         | 2020 |
| 34   | GAIN-GloVe                         | 55.08 | [Double Graph Based Reasoning for  Document-level Relation Extraction](https://arxiv.org/pdf/2009.13752v1.pdf) | [code](https://github.com/DreamInvoker/GAIN)       | 2020 |
| 35   | LSR+GloVe                          | 54.18 | [Reasoning with Latent Structure  Refinement for Document-Level Relation Extraction](https://arxiv.org/pdf/2005.06312v3.pdf) | [code](https://github.com/nanguoshun/LSR)          | 2020 |
| 36   | Two-Step+BERT-base                 | 53.92 | [Fine-tune Bert for DocRED with Two-step  Process](https://arxiv.org/pdf/1909.11898v1.pdf) | [Code](https://github.com/hongwang600/DocRed)      | 2019 |
| 38   | BERT-base                          | 53.2  | [Fine-tune Bert for DocRED with Two-step  Process](https://arxiv.org/pdf/1909.11898v1.pdf) | [code](https://github.com/hongwang600/DocRed)      | 2019 |
| 40   | DocRED-BiLSTM                      | 51.06 | [DocRED: A Large-Scale Document-Level  Relation Extraction Dataset](https://arxiv.org/pdf/1906.06127v3.pdf) | [code](https://arxiv.org/pdf/1906.06127v3.pdf)     | 2019 |



## 模型方法：

### 有监督：

主要分为pipeline 和 joint。

#### pipeline：

**早年神经网络：**

【1】Position-aware Attention and Supervised Data Improve Slot Filling [【论文】](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fpubs%2Fzhang2017tacred.pdf#=&zoom=140)

【2】 Relation Classification via Convolutional Deep Neural Network[[ 论文 ]](https://www.aclweb.org/anthology/C14-1220/)

![](https://i.loli.net/2019/12/06/YWIB2QDk38caJn7.png)

该模型将关系抽取任务利用神经网络进行建模，利用无监督的词向量以及位置向量作为模型的主要输入特征，一定程度上避免了传统方法中的误差累积。但仍然有 lexical level feature 这个人工构造的特征，且 CNN 中的卷积核大小是固定的，抽取到的特征十分单一.                                                                                                                                                                   https://www.aclweb.org/anthology/C14-1220/)



#### 依存信息

**基础做法：**用工具获取对应的依存信息，再采用一定的修剪策略，利用神经网络引入，[1],[2],[3]分别为不同的策略。

[1] SDP 最短依存路径：Classifying Relations via Long Short Term Memory Networks along Shortest Dependency Paths [[论文]](https://arxiv.org/abs/1508.03720)[代码]

[2] SP-Tree: 最小公共祖先：End-to-end relation extraction using lstms on sequences and tree structures [论文 ](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F1601.00770.pdf)[代码]

[3]修剪后用GCN 加载依存信息： Graph convolution over pruned dependency trees improves relation extraction [【论文】](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1809.10185.pdf) [【代码】](https://link.zhihu.com/?target=https%3A//github.com/qipeng/gcn-over-pruned-trees)

[4] 不修剪+multi head attention 的GCN加载依存信息，实现自动选择重点：Attention Guided Graph Convolutional Networks for Relation Extraction [ 【论文】 ](https://link.zhihu.com/?target=https%3A//www.aclweb.org/anthology/P19-1024.pdf)  [ 【代码】 ](https://link.zhihu.com/?target=https%3A//github.com/Cartus/AGGCN)

[6] dynamic 修剪: Learning toPrune Dependency Trees with Rethinking for Neu-ral Relation Extraction. [论文](https://www.aclweb.org/anthology/2020.coling-main.341/)  

[7]additional attention layers:   Graph Convolution over Multiple De-pendency Sub-graphs for Relation Extraction. [论文](https://xueshu.baidu.com/usercenter/paper/show?paperid=1v7c0mt0pg2e0xa0v71t06f0n1538687) [[代码]]()

[8] 结合修剪策略+依存关系类型A-GCN:
Dependency-driven Relation Extraction with Attentive Graph Convolutional Networks  [论文](https://aclanthology.org/2021.acl-long.344.pdf)
简介： 先获取依存类型的embedding词表，与encoder后的词向量拼接，双双做点乘，增加双方的交互，作为GCN的邻接矩阵。
![AGCN](/Users/lilili/Desktop/毕设/paper_pic/AGCN.png)
其中pij的获取方式：si和sj先内积再softmax得到pij

[9] 依存类型之TaMM: 

	（1）tamm方式加载依存树
	（2）获取两种memory slots方式：（1）与自己直接相连的 （2）到第二个实体的路径。
	（3）只使用实体的信息

[10] Relation Extractionwith Convolutional Network over Learnable Syntax-Transport Graph 【[论文](https://ojs.aaai.org/index.php/AAAI/article/view/6423)】【代码】

#### 联合实体训练

1. 多任务学习（共享参数）：

   【1】构建 N\*N\*C的关系分类器对每一个实体pair进行关系预测（N为序列长度，C为关系类别总数），输入的实体pair其实是每一个抽取实体的最后一个token。后续基于多头选择机制，也有paper引入预训练语言模型和bilinear分类。

2. 结构化预测：



【1】GrantRel: Grant Information Extraction via Joint Entity and Relation Extraction

【2】Effective Cascade Dual-Decoder Model for Joint Entity and Relation Extraction

【3】Injecting Knowledge Base Information into End-to-End Joint Entity and Relation Extraction and Coreference Resolution [论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2010.03851.pdf#=&zoom=140)

【4】joint_GCN（ACL2019）：Joint Type Inference on Entities and Relations via Graph Convolutional Networks [论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2FP19-1131.pdf#=&zoom=140) [代码](https://github.com/changzhisun/AntNRE)

【5】emnlp2020：2better1：Two are Better than One:Joint Entity and Relation Extraction with Table-Sequence Encoders [论文](https://arxiv.org/abs/2010.03851) [代码](https://link.zhihu.com/?target=https%3A//github.com/LorrinWWW/two-are-better-than-one)

【6】A Frustratingly Easy Approach for Joint Entity and Relation Extraction [论文](https://arxiv.org/abs/1804.07847)

#### 事件检索联合

【1】OneIE （ACL2020）： A Joint Neural Model for Information Extraction with Global Features [论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2F2020.acl-main.713.pdf#=&zoom=140) [代码](http://blender.cs.illinois.edu/software/oneie)

【2】ijcai2020: Attention as Relation: Learning Supervised Multi-head Self-Attentionfor Relation Extraction[论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.ijcai.org%2FProceedings%2F2020%2F0524.pdf#=&zoom=140)

### 远程监督：

数据集DocRED：Docred:   A  large-scaledocument-level relation extraction dataset.[【论文】](https://arxiv.org/abs/1906.06127)

【1】three-sentence：**Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction**[【论文】](https://arxiv.org/abs/2106.01793)[【代码】]( https://github.com/AndrewZhe/ThreeSentences-Are-All-You-Need.)

【2】LSR： Reasoning with latent structure refifine ment for document-level relation extraction. [【论文】](https://doi.org/10.18653/v1/2020.acl-main.141)[【代码】](https://github.com/nanguoshun/LSR)

【3】GCNN：Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network [【论文】](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fpubs%2Fzhang2017tacred.pdf#=&zoom=140)【代码】

【4】GraphRel：**GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction** 【[论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2F2020.acl-main.141.pdf#=&zoom=140)】【代码】

【5】AAAI2020:Neural Relation Extraction within and across Sentence Boundaries [论文](https://ojs.aaai.org/index.php/AAAI/article/view/4617)



### 具体论文f1值：

|                |       |         |        | 远程监督 | 文档级别 | fewshot    |
| -------------- | ----- | ------- | ------ | -------- | -------- | ---------- |
|                | ACE   | semeval | tacred | NYT      | DocRED   | FewRel     |
| RNN_2015       |       | 79.6    |        |          |          |            |
| CNN-PE         |       |         | 61.2   |          |          |            |
| SDP-LSTM       |       | 83.7    | 58.7   |          |          |            |
| att_lstm_2016  |       | 84      |        |          |          |            |
| position_2017  |       |         | 66     |          |          |            |
| C-GCN          |       | 84.8    | 68.2   |          |          |            |
| AGGCN          | -     | 85.4    | 68.2   |          | 51.45    |            |
| PRE            | -     |         | 68.41  |          |          |            |
| dynamic        |       | 86.4    | 69.2   |          |          |            |
| sub-graphs     |       | 85.9    | 66.1   |          |          |            |
| A_GCN          | 79.05 | 89.85   |        |          |          |            |
| TaMM           | 78.98 | 90.06   |        |          |          |            |
| three-sentence |       |         |        |          | 56.23    |            |
| LSR            |       |         |        |          | 59.05    | 54.18glove |
| GCNN           |       |         |        |          | 51.62    |            |
| GraphRel       | 57.72 |         |        | 54.9     |          |            |
| overlap        | 62.02 | 88.96   |        | 56.47    |          |            |



