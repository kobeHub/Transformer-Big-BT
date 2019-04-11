# Transformer 模型概述

[TOC]

## 1. 背景

   Seq2seq 模型是机器翻译，对话生成等任务中的经典模型，attention 机制为其提供了较大的性能改进。基本的Seq2seq 模型使用的是复杂的RNN或者CNN作为`encode，decoder`，加上attention机制后为模型赋予了较强的区分辨别能力。而 `Transformer`模型完全使用attention机制摒弃了RNN，CNN。建立了更为简单的模型，得到了优异的性能，更易并行化，训练时间更少。原始的 `Transformer` 模型在WMT2014 英德数据集上取得了28.4BLEU的性能。

   之后还有很多基于`Transformer`模型的改进,本此报告先总结`Transformer`原始模型的基本思想以及算法。

## 2. Attention Mechanism

### 2.1 原理

Seq2seq 模型解决的主要问题是如何把一个变长的输入$x$映射到一个变长的输出$y$.基本的结构如下图所示：
![seq2seq](https://user-gold-cdn.xitu.io/2017/11/30/1600b8156d71e47d?imageslim)

其中Encoder把输入的$T$维向量编码为一个固定长度的隐向量（或者成为上下文向量context），

