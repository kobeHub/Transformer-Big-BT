# 机器翻译开题报告

[TOC]

## 1. 实验目标

   此次实验的目标是基于现有研究成果，实现一个中英互译的机器翻译引擎，可以提供中英互译的基本API。利用实验提供的`UMCORPUS` 数据集以及其他的一些开源数据集（详情见下文）进行训练。根据 [nlpprogress](<http://nlpprogress.com/english/machine_translation.html>) 的记录，目前在机器翻译领域（基于WMT2014数据集）表现最为优秀的模型是[Transformer Big + BT (Edunov et al., 2018)](https://arxiv.org/pdf/1808.09381.pdf)(英德互译)，[DeepL](https://www.deepl.com/press.html)(英法互译)，以及表现同样较为优秀的 [DynamicConv (Wu et al., 2019)](https://arxiv.org/abs/1901.10430) 等。其中DeepL是商业软件，其他两个都有论文发表。实验的基本思路是借鉴这些优秀论文的思想，比较具体的可行性，选取其中的一个进行实现。

从机器翻译的基本方法出发，本次实验选取了能够代表三大方法的开源 ML 引擎进行对比。包含但不限于以下模型：

+ [Apertium](https://github.com/apertium)  基于规则的
+ [Moses](http://www.statmt.org/moses/?n=Moses.Overview)     基于统计的
+ [THUMT](<http://thumt.thunlp.org/>)     基于神经网络的

对于以上模型进行核心模型以及算法的分析，并且进行验证分析，与自己实现的模型进行对比。

实验的最终目标是通过对于机器翻译的模型的搭建，对于不同方法的比较验证，理解和掌握有关机器翻译的经典方法，同时对一些state of the art的方法有一定的了解。

## 2. 模型实现的基本思路

   从 MT 基于规则的方法，到依赖于大量数据的基于统计的方法，再到基于深度学习的 `seq2seq learning`。机器翻译领域不断涌现着新的思路和方法，并且表现出更优异的性能。从`seq2seq`的思路出发，机器翻译模型通过将一种语言的输入序列转化为另一种语言的输出序列，从而达到翻译的目的。基本的模型由两个 RNN 组成，一个`encoder`，一个`decoder`。由于RNN具有保存先前状态的能力，所以可以学习序列化数据的规律。之后在`seq2seq`的方法的基础之上，提出了`attention`机制，使得模型性能具有了较大提升。之后又提出的 [Transformer(Attention is all you need)](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1706.03762) 不再使用RNN作为`encoder`以及`decoder`，而是全部使用`attention`。建立了直接的长距离依赖。

   所以现有的基本思路是建立Transformer的模型，通过研读有关论文，在此基础上使用Back-Translation的方法。最终目标是最大近似的再现 [Transformer Big + BT (Edunov et al., 2018)](https://arxiv.org/pdf/1808.09381.pdf) 的模型。有关模型的细节以及具体实施方案，会出现在后续的报告中。

## 3. 语料库汇总

在给定的语料库基础之上，又选取了一些开源的中英平行语料库进行扩充，具体语料库如下：

+ [WMT18](<http://statmt.org/wmt18/translation-task.html#download>):

  + [[News Commentary v13](http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz)] 111M
  + [CWMT Corpus](http://nlp.nju.edu.cn/cwmt-wmt/)
    + casia2015.zip  94M
    + casict2015.zip 95M
    + datum2017.zip  102M
+ [OpenSubtitles](<http://opus.nlpl.eu/OpenSubtitles2016.php>)   304M
+ [MultiUN](<http://opus.nlpl.eu/MultiUN.php>)         859M
+ UMCORPUS        211M

![dir](/home/inno/Pictures/190407-dic.png)

*total: 1.8G*

## 4. 环境

### 4.1 开发环境

+ 操作系统： Arch Linux
+ 编辑器：   vim

### 4.2 运行环境

+ Docker: 18.09.4-ce
+ docker image: tensorflow/tensorflow:1.12.0-gpu-py3
+ cuda: 9.0
+ cudnn: 7.1
+ tensorflow-1.12.0
+ keras-2.2.4
+ numpy-1.15.4
+ scipy-1.1.0
+ scikit-learn0.20.0
+ scikit-image0.12.3
+ torch-1.0.0









