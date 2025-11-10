# 编译指令

```shell
nvcc -G -g -arch=sm_80 -o mtn mtn.cu
```
- `-G`: 生成CUDA调试信息
- `-g`: 生成HOST调试信息
- `-arch=sm_80`: 指定CUDA架构为sm_80 (Ampere 架构)

# 常用指令 （cuda-gdb help）
| 指令          | 简写 | 指令说明                                                                 | 举例                              |
|---------------|------|--------------------------------------------------------------------------|-----------------------------------|
| file exe_name |      | 指定待调试的可执行文件                                                   | `file program`                    |
| set args      |      | 设置命令行参数                                                           | `set args 1 2`                    |
| breakpoint    | b    | 设置断点                                                                 | `b main`<br>`b 数字`              |
| run           | r    | 在调试器中执行程序                                                       |                                   |
| start         |      | 开始执行程序，并在main的第一行停住                                       |                                   |
| next          | n    | 单步执行到下一行                                                         |                                   |
| step          | s    | 单步执行，会进入函数内部执行                                             |                                   |
| continue      | c    | 执行已暂停程序到下一断点或结尾处                                         |                                   |
| print         | p    | 打印参数信息，查看变量                                                   | `p var1`                          |
| thread        |      | 列出当前主机线程                                                         |                                   |
| cuda          |      | 列出当前活跃的kernel/grid/block/thread内容，并允许将焦点移至此处         | `cuda thread(1,1,1)`<br>`cuda kernel 1 block(1,2,1)` |
| info          |      | 查看参数所包含的具体信息                                                 | `info devices`<br>`info kernels`<br>`info threads` |
| backtrace     | bt   | 显示当前函数调用栈的内容                                                 |                                   |

> **勘误说明**
> 原始数据中 `next` 指令简写与 `run` 冲突，根据通用调试器规范修正为 `n`

# 官方链接
- [官方链接](https://docs.nvidia.com/cuda/cuda-gdb/index.html#cuda-gdb-extensions)