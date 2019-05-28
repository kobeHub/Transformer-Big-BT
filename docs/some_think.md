## 1. transformer 模型的优势?

+ Transformer设计的最大的带来性能提升的关键在于将任意两个单词间的距离作为1进行计算，对于解决长期依赖较为有效。可以建立直接的长期依赖
+ 完全使用Attention机制，模型的设计十分精彩，为后续的工作提供了极大地启发
+ 由于没有使用RNN，所以可以大大降低每一层的计算复杂度，可以并行计算

## 2. Transformer缺点？

+ 粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力
+ Transformer失去的位置信息其实在NLP中非常重要，而论文中在特征向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。
+ RNN + CNN + Transformer的结合可能会带来更好的效果
+  Transformer可以使模型进行并行训练, 但是仍然是一个autoregressive的模型; 也就是说, 任意一帧的输出都是要依赖于它之前的所有输出的. 这就使得inference的过程仍然是一个sequential procedure. 