# 0 MOE 概述

- [Gshard](https://arxiv.org/pdf/2006.16668)
- [Switch-Transformer](https://arxiv.org/pdf/2101.03961)

当你查看最新发布的大型语言模型（LLM）时，经常会在标题中看到“MoE”。这个“MoE”代表什么？为什么这么多LLM都在使用它？

在本视觉指南中，我们将通过50多个可视化图表，慢慢来探索这个重要的组件——混合专家（MoE）。

![MOE Simple](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F50a9eba8-8490-4959-8cda-f0855af65d67_1360x972.png)

在本视觉指南中，我们将介绍MoE的两个主要组件，即在典型的基于LLM的架构中应用的专家（Experts）和路由器（Router）。<br>

# 1 What is Mixture of Experts
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;混合专家（MoE）是一种技术，它使用许多不同的子模型（或“专家”）来提高大型语言模型（LLM）的质量。<br>

**MoE由两个主要组件定义：** <br>

专家（Experts）- 现在，每个前馈神经网络（FFNN）层都有一组“专家”，可以选择其中的一个子集。这些**专家本身通常是前馈神经网络**。

路由器或门控网络（Router or gate network）- 决定哪些词元**tokens被发送到哪些专家**。

在具有MoE的LLM的**每一层**中，我们找到（某种程度上专门化的）专家：

![Multi-layers experts](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7931367a-a4a0-47ac-b363-62907cd6291c_1460x356.png)

要知道，“专家”并不是专注于像“心理学”或“生物学”这样的特定领域。至多，它是在单词层面上学习句法信息: <br>

![Layer1 expert](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc6a81780-27c8-45f8-bccc-cc8f1ce3e943_1460x252.png)


更具体地说，它们的专长是在特定上下文中处理特定词元。

路由器（门控网络）选择最适合给定输入的专家：<br>

![figure 4](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb6a623a4-fdbc-4abf-883b-3c2679b4ad4d_1460x640.png)


每个专家并不是一个完整的LLM，而是LLM架构中的一个子模型部分。

# 2 The Experts
为了探索专家所代表的含义及其工作原理，让我们首先来考察一下MoE旨在替代的是什么：密集层(Dense Layer)。<br>

## 2.1 Dense Layer

混合专家（MoE）都始于大型语言模型（LLM）的一个相对基本的功能，即前馈神经网络（FFNN）。

请记住，在仅解码器的标准Transformer架构中，前馈神经网络是在层归一化(layernorm)之后应用的：<br>

![figure 5](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd4729d2a-a51a-4224-93fe-c5674b9b38eb_1460x800.png)

前馈神经网络（FFNN）使模型能够利用注意力机制生成的上下文信息，并对其进行进一步转换，以捕捉数据中更复杂的关系。

然而，前馈神经网络的规模会迅速增长。为了学习这些复杂的关系，它通常会扩展其接收的输入：


![figure 6](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F091ec102-45f0-4456-9e0a-7218a49e01df_1460x732.png)

## 2.2 Sparse Layer
传统Transformer中的前馈神经网络（FFNN）被称为密集模型，因为所有参数（其权重和偏置）都被激活。没有任何参数被遗漏，所有参数都用于计算输出。<br>

如果我们仔细观察这个密集模型，会发现**输入在某种程度上激活了所有参数**：<br>

![figure7](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F101e8ddc-9aa7-4e24-92fc-78d25da73399_880x656.png)


相比之下，稀疏模型只激活其总参数的一部分，并且与混合专家（Mixture of Experts）密切相关。

为了说明这一点，我们可以将密集模型切成小块（即所谓的专家），重新训练它，并在给定时间只激活一部分专家：<br>

![figure 8](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc4eeaf8-166b-419f-896c-463498af5692_880x656.png)

其基本思想是，每个专家在训练过程中学习不同的信息。然后，在进行推理时，只使用特定专家，因为它们与给定任务最相关。

当被问到一个问题时，我们可以选择最适合给定任务的专家：<br>

![figure 9](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fce63e5cc-9b82-45b4-b3dc-9db0cac47da3_880x748.png)

# 3 What does an Expert Learn
正如我们之前所见，专家学习的是比整个领域更加细粒度的信息([论文链接](https://arxiv.org/pdf/2202.08906))。因此，有时将它们称为“专家”可能会被视为具有误导性。<br>

**Expert specialization of an encoder model in the ST-MoE paper.** <br>

![figure 10](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F04123f9e-b798-4712-bcfb-70a26438f3b9_2240x1588.png)


然而，解码器模型中的专家似乎并不具备同样类型的专业化(specialization)。不过，这并不意味着所有专家都是等同的。

在Mixtral 8x7B论文中可以找到一个很好的例子，其中每个标记都用首选专家的颜色来标识。

![figure 11](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd03e32b4-5830-4d98-8514-0c1a28127ed9_1028x420.png)

这个可视化还表明，**专家往往更关注语法而不是特定领域**。<br>

因此，尽管解码器专家似乎没有特定的专长，但它们似乎在某些类型的标记上被一致地使用。

# 4 The Architecture of Experts
虽然将专家可视化为一个被切割成多块的密集模型的隐藏层是很不错的，但它们本身往往就是完整的全连接前馈神经网络（FFNNs）：<br>

![figure 12](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe51561eb-f3d6-45ca-a2f8-c71abfa7c2a9_880x748.png)

由于大多数大型语言模型（LLMs）包含多个解码器块，因此在生成文本之前，给定的文本会经过多个专家处理：<br>

![figure 13](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F89b1caad-5201-43fe-b7de-04ebe877eb2d_1196x836.png)

所选的专家可能在不同的标记之间有所不同，这会导致走不同的“路径”：<br>

![figure 14](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcde4794d-8b3e-454d-9a1c-88c1999fdd45_1372x932.png)

如果我们更新解码器块的可视化，现在它会包含更多的全连接前馈神经网络（每个专家一个），如下所示：<br>

![figure 15](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb97a8ac7-db97-497f-866d-10400729d51e_1248x764.png)

解码器块现在拥有多个全连接前馈神经网络（每个都是一个“专家”），在推理过程中可以使用。<br>

# 5 The Routing Mechanism
现在我们有了一组专家，模型是如何知道该使用哪个专家的呢？

在专家之前，会添加一个路由器（也称为门控网络），它经过训练，能够为给定的标记选择合适的专家。

## 5.1 The Router
路由器（或门控网络）也是一个全连接前馈神经网络（FFNN），它根据特定输入来选择专家。它输出概率，并使用这些概率来选择最匹配的专家：<br>

![figure 16](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Facc49abf-bc55-45fd-9697-99c9434087d0_864x916.png)

专家层返回所选专家的输出(output)，该输出乘以门控值（selection probabilities）。

路由器(The router)与专家（其中只选择少数几个）一起构成了混合专家（MoE）层：<br>

![figure 17](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa6fcabc6-78cd-477f-ac4e-2260cb06e230_1160x688.png)

一个给定的混合专家（MoE）层有两种规模，即稀疏专家混合或**密集专家混合(工程上基本不用)**。

两者都使用路由器来选择专家，但稀疏MoE只选择少数几个专家，而密集MoE则选择所有专家，但可能以不同的分布进行选择。<br>

![figure 18](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F46aadf17-3afe-4c98-b57c-83b7b38918b2_1004x720.png)

例如，给定一组标记，一个混合专家（MoE）层会将这些标记分配给所有专家，而一个稀疏MoE层则只会选择少数几个专家。

在当前的大型语言模型（LLM）状态下，当你看到“MoE”时，它**通常指的是稀疏MoE**，因为它允许你使用专家的一个子集。这在计算上更便宜，这是LLM的一个重要特性。


## 5.2 Selection of Experts
门控网络可以说是任何混合专家（MoE）系统中最重要的组件，因为它不仅决定了在推理过程中选择哪些专家，还决定了在训练过程中选择哪些专家。

在其最基本的形式中，我们将输入（x）与路由权重矩阵（W）相乘：<br>

![figure 19](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58234ce0-bf96-49ab-b414-674a710a1c3c_1164x368.png)

然后，我们对输出应用SoftMax函数，为每个专家生成一个概率分布G(x)：<br>

![figure 20](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb888a32f-acef-4fff-9d4b-cc70e148a8f2_1164x384.png)


路由器使用这个概率分布来为给定的输入选择最匹配的专家。

最后，我们将每个路由器的输出与每个选定的专家的输出相乘，并将结果相加。

![figure 21](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe6e46ea4-dbd4-4cc4-aa2b-2c5474917f31_1164x464.png)

让我们把所有内容整合起来，探索一下输入如何通过路由器和专家进行流动：

![figure 22](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd5d24a0b-2d78-4c69-b6fe-d75ba34bdd0c_2080x2240.png)

![figure 23](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F3d1122aa-7248-47d0-8e01-caa941ce0aa9_2080x2240.png)

## 5.3 The Complexity of Routing
然而，这个简单的函数往往会导致路由器选择相同的专家，因为某些专家可能会比其他专家学习得更快：<br>

![figure 24](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F9233733c-c152-428a-ae99-1ed185fc3d50_1164x660.png)

这不仅会导致所选**专家的分布不均**，而且某些专家几乎**得不到任何训练**。这会在训练和推理过程中都引发问题。

相反，我们希望在训练和推理过程中`专家之间具有同等的重要性`，这称为**负载均衡**。在某种程度上，这是为了防止对同一组专家过度拟合。

# 6 Load Balancing
为了平衡专家的重要性，我们需要关注**路由器**，因为它是决定在给定时刻选择哪些专家的主要组件。

## 6.1 KeepTopK
路由器负载均衡的一种方法是通过一个名为[KeepTopK](https://arxiv.org/pdf/1701.06538)的简单扩展来实现。通过引入可训练的（高斯）噪声，我们可以防止总是选择相同的专家：<br>

![figure 25](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1b95b020-ae34-40f0-a5c4-9542343beea9_1164x412.png)

然后，除了你希望激活的前k个专家（例如2个）之外，其他所有专家的权重都将设置为-∞：<br>

![figure 26](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F66bea40e-3fb0-4937-88d5-2852af456cf3_1164x488.png)

通过将这些权重设置为-∞，对这些权重应用SoftMax函数后，得到的概率将为0：

![figure 27](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F687d2279-1d8b-4af1-b55e-55d618ee877f_1164x496.png)

KeepTopK策略是许多大型语言模型（LLM）至今仍在使用的一种策略，尽管有许多有前景的替代方案。请注意，KeepTopK**也可以在不添加额外噪声的情况下使用**。

## 6.2 Token Choice

KeepTopK策略将每个token路由到少数选定的专家。这种方法称为“[标记选择](https://arxiv.org/pdf/1701.06538)”，它允许将给定标记发送到一个专家（即top-1路由）：

![figure 28](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf7a9988-d4c8-4b1b-a968-073a6b3bfc6a_1004x648.png)

或者多于1个的路由:

![figure 29](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fb3f283f1-c359-4baf-8d01-8ebb2a90665f_1004x720.png)

一个主要的好处是，它能够对各专家的贡献进行权衡和整合。

## 6.3 辅助损失(Auxiliary Loss)
为了在训练过程中获得更均匀的专家分布，在网络的常规损失中添加了辅助损失（也称为**负载均衡损失**）。

它增加了一个约束，**强制要求专家具有同等的重要性**。

这个辅助损失的第一个组成部分是对整个批次中每个专家的路由器值进行求和：

![figure 30](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff3624da0-3137-42ba-95e8-88fcbddb5f9f_1108x288.png)

这为我们提供了每个专家的重要性得分，表示无论输入如何，选择给定专家的可能性有多大。

我们可以利用这一点来计算变异系数-coefficient variation（CV），它告诉我们专家之间的重要性得分差异有多大。

![figure 31](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F94def8dc-2a65-4a02-855f-219f0df2a119_916x128.png)

例如，如果重要性得分差异很大，那么变异系数（CV）就会很高：

![figure 32](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fab71b90c-ba29-42a9-944b-3dee52fc5c32_916x372.png)

相反，如果所有专家的重要性得分相似，那么变异系数（CV）就会较低（这是我们的目标）：

![figure 33](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fc5cb91ac-4aab-4eb5-80bf-84e2bd4dc576_916x324.png)


使用这个变异系数（CV）得分，我们可以在训练过程中更新辅助损失，使其旨在尽可能降低CV得分（从而使每个专家具有同等的重要性）：

![figure 34](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4aac801-af89-44e7-aaea-c57a55ff282c_916x312.png)

最后，辅助损失作为训练过程中要优化的一个独立损失被添加进来。


## 6.4 Expert Capcity (专家容量)

不平衡不仅体现在所选专家上，还体现在发送给专家的标记分布中。

例如，如果输入标记不成比例地发送给某一个专家而不是另一个专家，那么这也可能导致训练不足：

![figure 35](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F749eac8e-36e5-450f-a6fc-fbe48b7a1312_1004x484.png)

在这里，问题不仅仅在于使用了哪些专家，还在于它们**被使用的程度**。

解决这个问题的一个方法是**限制给定专家可以处理的标记数量**，即[专家容量:GShard](https://arxiv.org/pdf/2006.16668)。当`某个专家达到容量限制时，后续的标记将被发送给下一个专家`：

![figure 36](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fdf67563f-755a-47a7-bebc-c1ac81a01f8f_1004x568.png)

如果两个专家都达到了他们的容量限制，那么该标记将不会被任何专家处理，而是被发送到下一层。这被称为**标记溢出**。

![figure 37](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe92ce4c5-affa-454d-8fd2-4debf9a08ce2_1004x544.png)

# 7 Simplifying MoE with the Switch Transformer
首个处理混合专家（MoE）模型训练不稳定性问题（如负载均衡）的基于Transformer的MoE模型之一是[Switch Transformer]()。它简化了大部分架构和训练过程，同时提高了训练的稳定性。

## 7.1 The Switching Layer
Switch Transformer 是一种 T5 模型（编码器-解码器），它用切换层（Switching Layer）替换了传统的前馈神经网络（FFNN）层。切换层是一个稀疏的混合专家（MoE）层，它为**每个标记选择单个专家（即采用Top-1路由）**。

![figure 38](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F024d1788-9007-4953-9bf7-883da0db7f8d_1160x688.png)

路由器在计算选择哪个专家时并不采用特殊技巧，而是将输入与专家的权重相乘后取softmax值（**与之前的方法相同**）。

![figure 39](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff0758a7f-e26b-44b9-9d75-80ac6caa9802_1104x384.png)

这种架构（Top-1 路由）假设路由器只需要一个专家就能学会如何路由输入。这与我们之前看到的情况不同，在之前的情况下，我们假设标记应该被路由到多个专家（Top-k 路由）以学习路由行为。

## 7.2 容量因子(Capacity Factor)

容量因子是一个重要的数值，因为它决定了**专家能够处理的Token数量**。Switch Transformer 在此基础上进行了扩展，通过引入一个直接影响专家处理能力的容量因子。

![figure 40](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F22715139-3955-4e00-bed7-c45cffa52744_964x128.png)

专家处理能力的组成部分很简单：<br>

![figure 41](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Ff4b399c6-723b-4de6-94ca-7020cd1bb181_908x380.png)

如果我们增加容量因子，每个专家将能够处理更多的标记。

![figure 42](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F7fd2aea0-fddf-4a43-ac79-7c5e5194c115_1240x472.png)

然而，如果容量因子过大，我们会浪费计算资源。相反，如果容量因子过小，由于标记溢出，模型性能将会下降。

## 7.3辅助损失

为了进一步防止丢弃标记，引入了一种简化版的辅助损失。

这种简化损失不计算变异系数，而是根据每个专家分派的标记比例与路由器概率的比例来加权：

![figure 43](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F608da44a-7510-4ab6-97c9-e8ab212a567d_836x388.png)

由于目标是让标记在N个专家之间均匀路由，我们希望向量P和f的值都为1/N。

α是一个超参数，我们可以用来在训练过程中微调此损失的重要性。值过高会盖过主要损失函数的作用，而值过低则对负载均衡影响不大。

# 8 Active vs. Sparse Parameters with Mixtral 8x7B
混合专家（MoE）模型之所以引人入胜，很大程度上在于**其计算需求**。由于在任何给定时刻只使用一部分专家，因此我们**能够访问的参数数量比实际使用的要多**。

尽管给定的MoE模型需要加载更多的参数（稀疏参数），但在推理过程中由于我们只使用部分专家，因此实际激活的参数较少（活跃参数）。

![figure 44](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fe1fd47bb-9ced-42e4-8f6c-536f7a65fbf7_1376x1252.png)

换句话说，我们仍然需要将整个模型（包括所有专家）加载到您的设备上（稀疏参数），但在进行推理时，我们只需要使用其中的一部分（活跃参数）。混合专家模型需要更多的显存来加载所有专家，但在推理过程中运行得更快。

让我们以[Mixtral 8x7B]()为例，来探讨一下稀疏参数和活跃参数的数量。

![figure 45](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fcc3d48d5-8afc-4477-af98-5817b1a145ae_1376x988.png)

在这里，我们可以看到每个专家的大小是5.6B，而不是7B（尽管有8个专家）。

![figure 46](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1dfd20b4-d3b7-433b-8072-2e67fc70afaa_1376x544.png)

我们需要加载8x5.6B（46.7B）个参数（以及所有共享参数），但进行推理时只需要使用2x5.6B（12.8B）个参数(**推理时Token 是一逐个生成的**)。<br>

# 9 Mixtrial
- TODO

# 10 DeepSeeKMOE
- TODO

# 10 参考资料
- [A Visual Guide to Mixture of Experts (MoE)](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts)