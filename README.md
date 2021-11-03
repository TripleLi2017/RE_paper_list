#### 关系抽取论文list

#### 数据集： 

`

ACE

semeval

tacred

[DocRED](https://github.com/thunlp/DocRED)

NYT

`







#### 具体论文f1值：

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

#### 具体论文

#### 依存类别

基础做法，用工具获取对应的依存信息，再采用一定的修剪策略，利用神经网络引入，[1],[2],[3]分别为不同的策略。

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

#### 实体

【1】GrantRel: Grant Information Extraction via Joint Entity and Relation Extraction
【2】Effective Cascade Dual-Decoder Model for Joint Entity and Relation Extraction
【3】Injecting Knowledge Base Information into End-to-End Joint Entity and Relation Extraction and Coreference Resolution

【1】joint_GCN（ACL2019）：Joint Type Inference on Entities and Relations via Graph Convolutional Networks [论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2FP19-1131.pdf#=&zoom=140) [代码](https://github.com/changzhisun/AntNRE)

事件检索联合

【1】OneIE （ACL2020）： A Joint Neural Model for Information Extraction with Global Features [论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2F2020.acl-main.713.pdf#=&zoom=140) [代码](http://blender.cs.illinois.edu/software/oneie)

【2】ijcai2020: Attention as Relation: Learning Supervised Multi-head Self-Attentionfor Relation Extraction[论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fwww.ijcai.org%2FProceedings%2F2020%2F0524.pdf#=&zoom=140)

#### 文档级别



【1】three-sentence：**Three Sentences Are All You Need: Local Path Enhanced Document Relation Extraction**[【论文】](https://arxiv.org/abs/2106.01793)[【代码】]( https://github.com/AndrewZhe/ThreeSentences-Are-All-You-Need.)

【2】LSR： Reasoning with latent structure refifine ment for document-level relation extraction. [【论文】](https://doi.org/10.18653/v1/2020.acl-main.141)[【代码】](https://github.com/nanguoshun/LSR)

【3】GCNN：Inter-sentence Relation Extraction with Document-level Graph Convolutional Neural Network [【论文】](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fpubs%2Fzhang2017tacred.pdf#=&zoom=140)【代码】

【4】GraphRel：**GraphRel: Modeling Text as Relational Graphs for Joint Entity and Relation Extraction** 【[论文](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Faclanthology.org%2F2020.acl-main.141.pdf#=&zoom=140)】【代码】





2019年之前：

【1】Position-aware Attention and Supervised Data Improve Slot Filling [【论文】](chrome-extension://ikhdkkncnoglghljlkmcimlnlhkeamad/pdf-viewer/web/viewer.html?file=https%3A%2F%2Fnlp.stanford.edu%2Fpubs%2Fzhang2017tacred.pdf#=&zoom=140)

【2】 Relation Classification via Convolutional Deep Neural Network[[ 论文 ]](https://www.aclweb.org/anthology/C14-1220/)

![](https://i.loli.net/2019/12/06/YWIB2QDk38caJn7.png)

该模型将关系抽取任务利用神经网络进行建模，利用无监督的词向量以及位置向量作为模型的主要输入特征，一定程度上避免了传统方法中的误差累积。但仍然有 lexical level feature 这个人工构造的特征，且 CNN 中的卷积核大小是固定的，抽取到的特征十分单一.

