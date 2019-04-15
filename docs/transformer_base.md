# Transformer 模型概述

[TOC]

## 1. 背景

   Seq2seq 模型是机器翻译，对话生成等任务中的经典模型，attention 机制为其提供了较大的性能改进。基本的Seq2seq 模型使用的是复杂的RNN或者CNN作为`encode，decoder`，加上attention机制后为q模型赋予了较强的区分辨别能力。而 `Transformer`模型完全使用attention机制摒弃了RNN，CNN。建立了更为简单的模型，得到了优异的性能，更易并行化，训练时间更少。原始的 `Transformer` 模型在WMT2014 英德数据集上取得了28.4BLEU的性能。

   之后还有很多基于`Transformer`模型的改进,本此报告先总结`Transformer`原始模型的基本思想以及算法。

## 2. Attention Mechanism

### 2.1 原理

Seq2seq 模型解决的主要问题是如何把一个变长的输入$x$映射到一个变长的输出$y​$.基本的结构如下图所示：
![seq2seq](https://user-gold-cdn.xitu.io/2017/11/30/1600b8156d71e47d?imageslim)

其中Encoder把输入的$T$维向量编码为一个固定长度的隐向量c（或者成为上下文向量context），其作用有二：初始化Decoder模型，作为背景向量指导序列中每一步$y_t$的产出。Decoder主要通过每一步的背景向量以及上一步的状态$y_{t-1}$得到时刻t的输出$y_t$,直到序列结束（<EOS>）。

但是基础的Seq2seq模型对于输出序列x缺乏区分度，所以加入的Attention机制，下图是[Bahadanau attention](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.0473)的示意图:
![attention](https://user-gold-cdn.xitu.io/2017/11/30/1600b8156c44218a?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

该模型中定义了一个条件概率：
![](https://user-gold-cdn.xitu.io/2017/11/30/1600b8158dc4c272?imageView2/0/w/1280/h/960/format/webp/ignore-error/1)

通过attention机制，将输入序列$X =(x_0, x_1, ...,x_T)$ 映射到一个隐层状态$H=(h_0, h_1, ...,h_T)$,再由Decoder将隐层 $H$,映射到输出序列$Y=  (y_0, y_1, ...,y_t)$.这里的精妙之处在于，输出序列Y的每一个元素都与隐层状态H相连，而H又与所有的输入状态存在联系，所以可以直接建立长距离的依赖，发掘更多语义上的关联，从而达到更好的翻译效果。

### 2.2 结构

对于上图所示的模型，总结如下：

+ 使用$h$表示Encoder的隐含状态，$s​$表示Encoder 的隐含状态

+ Encoder将输入$X = (x_0, x_1, ...,x_{T_x})$,通过一个双向LSTM得到两组隐含状态向量$h^{\leftarrow}, h^{\rightarrow}$.然后连接起来得到最终的$H=(h_1,h_2, ...,h_{T_x})$.

+ 对于Decoder，在时刻$t$时一共有三个输入： $s_{t-1}, y_{t-1}, c_t$,分别代表上时刻的隐含状态、上一时刻的输出、当前步的上下文向量c（即所有Encoder output经过加权得到的一个定长的向量），作为Decoder的上下文。

+ 有关$c_t$的计算，需要使用softmax作为权重公式，$c_i = \sum_{j=1}^{T_x}a_{ij}h{j}$。其中$a_{ij}$是对应的权值，$h$是Encoder每一步的隐含状态(hidden state, value 或者memory)。i为Decoder step，j为Encoder step。通过这样的权重分配，给与当前上下文相关度较大的状态赋予大的权重，对于Decoder进行解码操作时的影响也就更大，也就是pay more attention

+ 对于计算$c_i$时使用的权重$a_{ij}$的公式如下:
  $$
  a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}{exp(e_{ik})}}
  $$
  使用的$e_{ij}$是根据某种度量条件计算得到的$s_{i-1},h_j$的相关程度，计算过程如下:

  > 1. 对于$s_{i-1}$做一个线性映射，得到的向量称为query，$q_i$
  > 2. 对于$h_i$做一个线性映射，得到的向量称为key, $k_j$
  > 3. $e_{ij} = v^T * (q_i + k_j)$; v是一个[d, 1]的向量，$q_i, k_j$ 的维度相同为 d

  以上步骤的线性映射以及v可以通过训练得到，这种方式称为**加性attention**。

+ 总结一下，attention就是算一个encoder output的加权和，叫做context；计算方法为，query和key计算相关程度，然后归一化得到alignment，即权重；然后用alignment和memory算加权和，得到的向量就是context。

### 2.3 Self-Attention

   Self Attention 与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐含状态（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。

​    但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关;捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系:

![self](/home/inno/Pictures/190413-att.png)

Transformer 模型即实现了self-attention.

## 3. Transformer

### 3.1 架构

Trandformer 将传统模型的Encode，Decoder都换为了多个attention。基本示意图如下：

![](https://pic1.zhimg.com/80/v2-26f3e9cd3679956dab33486c83dd0088_hd.jpg)

1. 左右分别是Encoder和Decoder
2. Encoder和Decoder的底部是embedding；而embedding又分为两部分：**input embedding**和**positional embedding**；其中**input embedding就是seq2seq中的embedding。**另一部分是positional embedding，添加该操作是由于transformer中只有attention；而对于attention机制，任意一对(query, key)的计算都是完全一样的，不像CNN和RNN，有一个位置或者时序的差异：CNN框住一块区域，随着卷积核移动，边缘的少量点会跟着有序变化；RNN更明显了，不同时序的 $h_t$ 和 $s_t$ 不同，而且是随着输入顺序不同（正序，倒序）而不同。因此为了体现出时序或者在序列中的位置差异，要对input加入一定的位置信息，即positional embedding。
3. Encoder 和 Decoder 分别是由`N=6`个相同的层叠加得到的。
4. 对于每一层，Encoder和Decoder的中部分别是两个block，分别输入一个序列、输出一个序列；Encoder的每个block里有两个子层，分别是MultiHead Attention和FeedForward Network; Decoder 的block里有三个子层，分别是两个MultiHead Attention和一个FFN。这些子网后面都跟了一个add&norm，并且仿照ResNet添加了一个`residual connection` ,对于每一个子层的输出可以形式化表示为`LayerNorm(x + SubLayer(x))` 也就是子层处理过后的结果加上输入在做正则化处理。
5. Decoder 的中间的　MultiHead Attention 接受Encoder的输出以及前一个Masked MultiHead Attention 的输出作为输入；
6. 其中Masked MultiHead Attention是修改过的self-attention, 由于输出的embedding具有一个位置的偏移量，使用masking确保了位置i的预测仅取决于小于i的位置处的已知输出。
7. Decoder最后还有一个Linear和Softmax。

### 3.2 MultiHead Attention

这是这篇论文的创新之处，对于原始的attention,就是一个*query*和一组*key*计算相似度，然后对于一组*value*计算加权和*output*。如果key，query的维度较高，比如均为512维向量，那么就需要在512维的高维空间比较两个向量的相似度。

而MultiHead Attention，则将原本的高维空间通过投影分为多个子空间，例如`head_num = 8`,那么就有8个子空间，相应的value也要分为8个head；然后分别在每一个子空间计算各自query， key的相似度，在分别与value结合。这样不仅降低了计算成本，便于并行处理，而且可以让attention从多个不同的角度进行结合，这对于NMT工作是很有帮助的。

因为处理翻译任务的source 与target不是一一对应的，而是由多个词共同决定的，进行不同的划分组合后，每一个子空间都可以从各自关注的角度取组合源语言，从而得到较好的翻译结果。

### 3.3 Feed-Forward Network  (FFN)

基本的前向反馈网络，使用全连接层即可实现。或者使用[1, 1]的卷积实现。使用ReLU作为激活函数。

![ffn](/home/inno/Pictures/190413-ffn.png)

### 3.4 Positional Embedding

由于网络中没有使用CNN，RNN，为了使用序列的位置信息，必须添加一些额外的信息确定一个token在序列中的相对或者绝对位置。文中使用的位置编码公式如下：
![pos](/home/inno/Pictures/190413-pos.png)

其中的$pos$是位置，$i$为维度。每一个维度的位置编码为一个正弦曲线，曲线的波长组成了范围在[2$\pi,10000*2\pi$]一个几何级数.使用这个函数，更容易通过相对位置进行学习，因为对于一个确定的偏差$k$,$PE_{pos+k}$可以使用$PE_{pos}$的线性方程表示。

## 4. Reference

Bahadanau attention:<https://arxiv.org/pdf/1409.0473.pdf>

Attention Is All You Need:<https://arxiv.org/pdf/1706.03762.pdf>

Understanding Back-Translation at Scale:<https://arxiv.org/pdf/1808.09381.pdf>

<https://zhuanlan.zhihu.com/p/38485843>

<https://juejin.im/entry/5a1f9e036fb9a0450671663c>

<https://medium.com/@adityathiruvengadam/transformer-architecture-attention-is-all-you-need-aeccd9f50d09>