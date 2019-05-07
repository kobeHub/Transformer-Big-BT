# 机器翻译中期报告

[TOC]

## 1. 数据预处理

### 1.1 分词与编码

​    对于 Transformer 的基本架构，使用基于字级别(英文基于word)的模型实现。使用平行语料库进行训练时，对于样本中的每一行$(source, target)$作为一个训练样本，这种方法的主要问题在于难以将具有长距离语义关联的词作为一个样本输入；但是如何发掘语义的关联程度从而确定每一个样本的长度，是一个较为繁琐的工作。所以打算使用一种折中的方式，先使用行粒度的较短样本进行预训练，然后使用多行长样本进一步训练。从而避免了过多预处理的工作。

   由于基于字符进行模型的实现，所以中文分词将每一个字作为最小单元，英文将每一个$word$作为最小单元，对于所有的训练、测试集进行遍历得到所需要的词库。除此之外，对于数据集中出现的标点符号，以及所有有意义的Unicode字符都需要加入到词库中。然后将词库中的所有$item$进行编号，对于每一个句子，可以用一个长度不定的一维向量表示。这样模型的输入就可以用两个一维向量表示。

```python
# 在Umcorpus中， 例句: We can tell from fossil evidence in rocks that many living species have become extinct over the millions of years since life began. 的向量表示为:

array([254939, 495884, 494380,  59409, 177392, 580889, 280523, 224280,
       388662, 460449,  24656, 493847, 572291, 407836, 554070, 285057,
       596466,   8485, 183316, 275391, 421102,  34097, 410489, 601320,
            1,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0,      0,      0,      0,      0,      0,      0,      0,
            0])
# 对应中文翻译
# 我们岩石中的化石遗留下的证据中可以辨别出来，自地球上有生命时开始，在若干个百万年中，有很多动植物的物种已经逐渐地灭绝了。

array([222842, 595004, 608916,  41411, 437579, 484813, 597251,  48875,
       437579, 425240, 305538, 116475, 597251, 567015,  32418, 484813,
       301587,  60117, 178244, 552052, 462251, 284775, 587092, 543810,
       608916,  27637, 191835, 456038, 317479, 209102,  19162, 559860,
       467261, 467056, 587092, 170418, 348904, 102539, 329258, 328805,
       216408, 399803, 484813, 587092, 317479,  40909, 371496, 173398,
       384795, 449458, 597251, 449458, 413263,   5051,   5334,  21244,
       266473,  27637,  27665, 163260, 368259, 587172,      1,      0,
            0,      0])
```

### 1.2 模型的输入

   为了减少IO时长，可以使用Tensorflow的数据格式TFRecord将数据集转换为二进制表示。通过实现建立的词库以及定义好的分词规则，将所有的训练集划分为100个TFRecord格式的输入样本，测试集作为一个TFRecord文件。进行训练集分割时，使用$Round-robin$ 的方法，同时配合$shuffle$操作，作为预处理的最终结果。对于UMcorpus数据集所有的预处理结果位于`~/data/UMcorpus/processed`,文件如下:

![](/home/inno/Pictures/190507-data.png)

### 1.3 Padding

由于输入句子的长度不一，所以模型输入的向量长度不同，对于同一个$batch$的句子需要找到最长的一个作为输入长度。其他的句子需要进行*padding*.为了尽量减少不必要的padding，所以需要将长度相近的输入样本作为一个$batch$。

*所有预处理工作代码在`~/code/preprocess/`文件夹下*

## 2. Transformer 模型的实现

### 2.1 Input Embedding

   定义`EmbeddingSharedWeights` 类，继承自`tf.layers.Layer`类，通过定义一个全连接层作为输入向量的embedding表示。定义一个单隐层的网络，共享weights的shape为`[vocab_size, hidden_size]`.这样对于一个shape为[batch_size, length]的input经过embedding后得到一个[batch_size, length, hidden_size]的向量。再配合*positional embedding*作为encoder，decode的输入。

​	由于Transformer实现的是一个**end-to-end**的模型，所以需要同时计算嵌入层的`logits`， 并向之后的层进行传递，作为计算最终logits的一部分，从而可以使用反向传播最小化loss。`logits`的计算使用的是$weights$, $inputs$ 的叉积。

*具体实现见`~/code/model/embedding_layers.py`*

### 2.2 Positional Embedding

根据论文中的描述，使用正弦曲线为位置编码。曲线的波长组成了范围在[2$\pi$, 10000*$2\pi$]的一个几何级数。使用这个函数，可以更容易的使用相对位置进行学习。

![](/home/inno/Pictures/190413-pos.png)

*具体实现见`~/code/model/model_utils.py`*

### 2.3 Multi-Headed Attention

定义`Attention`类，继承自`tf.layers.Layer`,该类中定义了三个全连接层:

+ `q_layer`: query 的线性映射
+ `k_layer`: key 的线性映射
+ `v_layer`: value 的线性映射

如果需要对于两个向量$x, y$使用attention机制：

+ 首先将$x$经过`q_layer`的线性映射得到`query`向量，将$y$经过`k_layer`,`v_layer`的映射得到`key`， `value`
+ 将得到的`query, key、value` 分为8个Head，从而使得每一个Head处理的向量的隐含维度从`hidden_size`下降到`hidden_size/8`.
+ 对于每一个Head，计算`query`,`key`的叉积作为logits，用于度量相似程度；然后计算logits的softmax值作为attention的权重
+ 计算`value`依据权重的加权和作为每一个Head的`attention_output`
+ 将8个Head合并为一个Head

### 2.4 Self-Attention

继承自Attention类，只需要将调用时的$x, y$参数替换为$x, x$.即可实现self-attention。

*具体实现见`~/code/model/attention.py`*

### 2.5 FeedForward Network Layer

继承自`tf.layers.Layer`类，是一个具有单隐含层的全连接网络。实际上做了一次非线性变换以及一次线性变换，同时对于输入数据使用padding的情况，在放入FFN处理前需要先去除padding，然后在对输出结果添加padding。

*具体实现见`~/code/model/ffn.py`*


### 2.6 EncoderStack

编码器Stack，包含了6个子层，每一个子层包含一个`self-attention`，一个`ffn`。同时`self-attention`,`ffn`的输出与各自输入的和作为一个子层的输出，从而实现类似ResNet的结构。**每一个子层的输出作为下一个子层的输入，最后一个子层的输出，经过一个BatchNormal层，作为整个EncoderStack的输出。**

*实现见`~/code/model/transformer.py`*

### 2.7 DecoderStack

解码器Stack，基本结构与`EncoderStack`类似，但是每一个子层中添加了一个`MultiHeaded layer` ,将 `EncoderStack` 的输出与`DecoderStack`的输入结合起来。

*实现见`~/code/model/transformer.py`*

****

**以上layers都默认添加了dropout,从而减低过拟合风险。**

****

```

最终的Transformer模型，使用EncoderStack，DecoderStack。将Embedding的输出作为输入(包含source、target)。
DecoderStack 的输出是一个softmax概率，代表了词库中每一个词出现在target翻译中的概率，此时根据输入target的向量建立起一个维度为vocab_size大小的one_hot向量，作为Transformer的label，从而计算交叉熵loss，再进行梯度下降最小化loss。

```

*有关损失函数的计算位于`~/code/train_and_evaluate.py`，`~/code/metrics.py`*



## 3. 模型train与eval

模型的train、eval、predict借助tensorflow的高级API Estimator 进行。对于一个Estimator,有三种用户可以调用的状态，`TRAIN`，`EVAL`，`PREDICT`。需要定义一个`model_fn`函数，用于为Estimator不同工作状态提供相应的指标以及操作。在该函数中，必须为每一种工作状态定义一个`tf.estimator.EstimatorSpec` 对象。

### 3.1 Learning deacy

为了降低过拟合风险，需要在模型训练到一定程度后降低学习率，采用**线性预热，steps平方根倒数式下降。**对于给定的超参数`learning_rate`,设置一个预热阈值，先乘以线性值，线性预热值，再乘以一个平方根倒数来降低学习率。

```python
def get_learning_rate(lr, hidden_size, lr_warmup_steps):
    with tf.name_scope('learning_rate'):
        warmup_steps = tf.cast(lr_warmup_steps, tf.float32)
        step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)

        lr *= (hidden_size ** -0.5)
        # Linear warmup
        lr *= tf.minimum(1., step / warmup_steps)
        # rsqrt deacy
        lr *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # name model/get_train_op/learning_rate/lr
        tf.identity(lr, 'lr')

        return lr
   
# 代码见 ~/code/train_and_eval.py
```

### 3.2 Training

由于Transformer类定义了`__call__`方法，所以当调用Transformer实例时，其执行`__call__()`方法。根据是否参数中是否有`target`向量判断模型处于train或者predict状态。将`inputs`输入进行`encode`得到的输出与`targets`一起作为`decode`的输入。代码框架如下:

```python
# Run inputs through encoder layers to map symbol representations
# to continuous representations
encoder_outputs = self.encode(inputs, attention_bias)

if target is None:
     return self.predict(encoder_output, attention_bias)
else:
     logits = self.decode(targets, encoder_outputs, attention_bias)
     return logits
```

对于输入的`inputs` shape 为`[batch_size, source_length]`,经过`Embedding`,`EncoderStack`,之后的shape为`[batch_size, source_length, hidden_size]`。`targets`输入时的shape为[batch_size, target_length],经过`Embedding`，shape=[batch_size, target_length, vocab_size]。再与`encode_output`一起作为`DecoderStack`的输入，最终的输出`shape=[batch_size, target_size, vocab_size]`。

### 3.3 Evaluate

对于设定的超参数，每隔一定的steps进行一次eval。主要的评价指标的定义位于`~/code/metrics.py`。同时计算`BLEU`,如果BLEU达到设定的最低下限，训练可以终止。

## 4. 现有进度

已经完成数据预处理以及格式化存储，Transformer模型搭建完成，基本的单元测试已经完成，还有部分Bug尚未修复。处于Debug阶段。

`Train`，`Translate`模块尚不完善，模型还不能跑通。 

模型对比方面，已经完成THUMT的分析，具体报告见其他文档。

## 5. Program Run

该项目一共有三个模块`pre_data`, `train`, `translate`，目前支持`pre_data`模块的正常使用，`train`模块正在调试中，`translate`模块在预训练模型完成之后方可使用。模块的参数如下:

### 5.1 pre_data

```
基本命令: python Main.py pre_data [args]
args:
     --raw_dir: 原始语料库的父文件夹，预处理模块遍历该文件夹找出所有的语料库文件，用于之后的数据处理
     --eval_dir: 测试数据所在的文件夹
     --data_dir: 生成的vocab，TFRecord的目标地址
     --shuffle: 是否对与生成的文件进行shuffle
```

### 5.2 train

```
基本命令：python Main.py train [args]
args:
	--bleu_source: 用于BLEU评分的源文件
	--bleu_ref:    用于BLEU评分的参考文件
	--num_gpus:    gpus 数量
	--data_dir:    预处理后的数据所在文件夹
	--model_dir:   生成的checkpoint的目标地址
	--export_dir:  Estimator 模型导出地址
	--batch_size:  指定的批处理数量
	--vocab_file:  语料库文件
	--stop_threshold:   停止训练的BLEU值
```

*所有参数已经赋予初始值，所以可以直接使用基本命令。*

### 5.3 translate

```
基本命令：python Main.py translate --text=/path/ | --inputs_file=/path/ --output_file=/path/
可以具有两种模式的翻译，可以通过--text参数进行句子的翻译，也可以翻译指定文件
```





