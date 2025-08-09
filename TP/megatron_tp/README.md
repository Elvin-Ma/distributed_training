# megatron-lm tp

![tp code implement](images/megatron-lm-tensor-parallel.drawio.png)


# 4 Megatron-lm 实现
```python
from megatron.core import mpu, tensor_parallel

mpu.initialize_model_parallel(args.tensor_model_parallel_size,
                  args.pipeline_model_parallel_size,
                  args.virtual_pipeline_model_parallel_size,
                  args.pipeline_model_parallel_split_rank)
```