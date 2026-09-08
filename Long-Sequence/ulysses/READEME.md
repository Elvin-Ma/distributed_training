# ulysses guide

Ulysses 天生就适合放在节点内——它的核心通信是 All-to-All,最吃对称的全互联带宽,NVLink 域正是理想场所。

核心思想是`转置`而非`流动`。

**Ring CP** vs **Ulysses**
- ring cp 让每卡守着自己的 token 段不动,靠 KV 绕环把"别人的 token"送过来;
- ulysses 利用的是"头与头之间完全独立"这一结构, attention 前仅把其他卡上对应头的 token 传过来即可;
- ulysses 在注意力前用 All-to-All 把数据从「序列切、头全」转置成「序列全、头切」;
- 转置之后,注意力变成每卡各自独立的标准单卡问题, 精确 softmax、FlashAttention 原样可用、causal 负载天然均衡;
- attention 算完再转置回去接 MLP。前向每层 **4 次** All-to-All(Q、K、V、O),反向镜像 **4 次**(dO、dQ、dK、dV).

# ulysses schematic diagram
# Ulysses schematic diagram

下面画 ulysses=4(占满节点 4 卡)的前向/反向分步原理图,布局上与前面的 CP 图对齐,便于对照:

[点击打开 Ulysses 交互式动画](./ulysses.html)

# ulysses comm volume

通信量随并行度反向缩放,这是 Ulysses 的账面优势。设序列长度为 $N$,隐藏维度为 $d$,并行度为 $P$。

单卡每次 All-to-All 的收发量约为：

$$
\left(\frac{N d}{P}\right)\frac{P-1}{P}
$$

每层前向和反向共 8 次 All-to-All,因此单卡每层通信量约为：

$$
8\frac{N d}{P}
$$

$P$ 越大,通信量越小。Ring CP 每层收发量约为 $2N d_{kv}$（反向再翻倍）,与 $P$ 无关。因此 MHA 大模型、$P$ 较大时 Ulysses 更省；而 GQA/MQA 下 $d_{kv}$ 较小,Ring 的通信量反而更低——两者恰好互补。

代价方面,Ulysses 有硬约束：$P$ 必须整除头数（GQA 下实际受 KV 头数限制,KV 头不足时需复制 KV 头,通信和显存都打折扣）,且 All-to-All 是全互联流量模式,放到跨节点的胖树/多轨网络上效率骤降——这正是“节点内进行即可”的原因：NVLink/NVSwitch 的对称全带宽是它的主场。

**MHA 通信量对比**

普通 MHA 中 $d_{kv}=d$。当 $P=8$ 时,Ulysses 单卡每层的精确通信量约为：

$$
8\times\left(\frac{Nd}{8}\right)\frac{7}{8}
=\frac{7}{8}Nd\approx Nd
$$

Ring CP 前向约为 $2Nd$,反向约为 $4Nd$,因此前向加反向约为 $6Nd$。

也就是说:<br>
- 只看前向时 Ring 约为 Ulysses 的 $16/7\approx2.3$ 倍；
- 前向加反向时约为 $48/7\approx6.9$ 倍。

因此在 $P=8$ 的普通 MHA 场景下,Ring CP 的通信量明显更大,Ulysses 更省通信。


# fsdp + ulysess

与外层 DP/FSDP 的组合逻辑和 CP 完全同构。

Ulysses 组内 4 卡处理同一条序列、参数互为副本,权重梯度各含本段 token 的贡献,组内还欠一次**求和**

和 FSDP + CP  的一模一样:
- 既可以把 ulysses 维放进 shard 维(FSDP 的 Reduce-Scatter 顺带完成求和,每卡常驻 1/4 分片),
- 也可以放在 replicate 维(靠跨副本 All-Reduce 完成)

mesh 写法形如:

```py
init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "ulysses"))
```

# USP（Unified Sequence Parallelism，统一序列并行）

- [paper](https://arxiv.org/abs/2405.07719)
- [code](https://github.com/feifeibear/long-context-attention)

本文中的 USP 指把 Ulysses 和 Ring Attention 组合起来的混合二维序列并行方法：节点内用 Ulysses（利用 NVLink 全互联、免改注意力内核）,跨节点用 Ring（只需邻居 P2P、可与计算流水重叠、不受头数限制）。

两种并行维度相乘得到更大的总序列并行度——例如 $4$（节点内 Ulysses）$\times$ $2$（跨节点 Ring）$=8$ 路序列并行。这样每个维度都工作在自己最适合的互联拓扑上；`LongContextAttention` 等实现也将这种方式称为 Unified-SP 或 Hybrid-SP。
