# 1 img2col

工程上卷积操作会被重新组织为矩阵乘法（GEMM）的形式，这一过程通常称为 im2col（Image to Column） 或类似的**内存重组技术**。以下是其核心原理和步骤：

1. 特征图展开（im2col）

将输入特征图（Input Feature Map）的每个局部窗口（对应卷积核的尺寸）展开为矩阵的一列.

例如：输入形状为 [N, C_in H, W]，卷积核大小为 [K, K]，则每个局部窗口的像素会被展开为 K*K*C_in 的列向量.

最终得到一个展开后的矩阵 **im2col_matrix**，形状为 [K*K*C_in, H_out*W_out*N]，其中 H_out 和 W_out 是输出的高和宽.

2. 卷积核展开

将卷积核（Filter）从 [C_out, C_in, K, K] 的形状展开为 [C_out, K*K*C_in] 的矩阵 **filter_matrix**.

3. 矩阵乘法

```python
output_matrix = filter_matrix @ im2col_matrix  # 形状为 [C_out, H_out*W_out*N]
```

4. reshape
最后将拼成输出的形状 [N, C_out, H_out, W_out]

# 2 example

1. 假设输入为 [1, 5, 5, 3]（Batch=1，高宽=5x5，通道=3），卷积核为 [3, 3, 3, 64]（64 个 3x3x3 的核）：

im2col 展开：

每个 3x3x3 的局部窗口展开为 27 维的列向量（3x3x3=27）。

输出特征图的每个位置对应一个窗口，总共有 3x3=9 个输出位置（假设 stride=1, padding=1）。

im2col_matrix 形状为 [27, 9]。

2. 卷积核展开：

64 个卷积核展开为 [64, 27] 的矩阵

3. 矩阵乘法：

output_matrix = [64, 27] @ [27, 9] = [64, 9]，最终重塑为 [1, 3, 3, 64]

# 3 Systolic Array
