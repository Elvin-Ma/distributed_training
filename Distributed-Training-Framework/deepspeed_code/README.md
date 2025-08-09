# 1. 启动训练脚本
## 1.1 训练脚本
```python
pip install deepspeed
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/training/cifar
bash run_ds.sh
```

## 1.2 相关环境依赖
- gcc 版本要小于 10;
- /usr/bin/gcc --> 链接到 /usr/bin/gcc-7 即可;

# 2 启动指令源码解读
```python
deepspeed --bind_cores_to_rank cifar10_deepspeed.py --deepspeed $@
```
其中: <br>
-  deepspeed 为可执行脚本：miniconda3/envs/pytorch2.0/bin/deepspeed
```python
#!/home/mtn/miniconda3/envs/pytorch2.0/bin/python
# EASY-INSTALL-DEV-SCRIPT: 'deepspeed==0.14.1+ffb53c25','deepspeed'
__requires__ = 'deepspeed==0.14.1+ffb53c25'
__import__('pkg_resources').require('deepspeed==0.14.1+ffb53c25')
__file__ = '/home/mtn/DeepSpeed/bin/deepspeed'
with open(__file__) as f:
    exec(compile(f.read(), __file__, 'exec'))
```

- 启动的文件为 /home/mtn/DeepSpeed/bin/deepspeed：
```python
#!/usr/bin/env python3

from deepspeed.launcher.runner import main

if __name__ == '__main__':
    main()

# main 中的只要工作
device_count = get_accelerator().device_count() # 设备数量
args.master_addr = "127.0.0.1" # 设定主机侧地址
active_resources = parse_inclusion_exclusion(resource_pool, args.include, args.exclude) # 过滤设备
world_info_base64 = encode_world_info(active_resources) # 编码激活的设备资源信息

multi_node_exec = args.force_multi or len(active_resources) > 1 # 是否多节点执行
# 接下来 为 launcher 准备命令行参数
if not multi_node_exec:
        deepspeed_launch = [
            sys.executable, "-u", "-m", "deepspeed.launcher.launch", f"--world_info={world_info_base64}",
            f"--master_addr={args.master_addr}", f"--master_port={args.master_port}"
        ]
        if args.no_python:
            deepspeed_launch.append("--no_python")
        if args.module:
            deepspeed_launch.append("--module")
        if args.no_local_rank:
            deepspeed_launch.append("--no_local_rank")
        if args.save_pid:
            deepspeed_launch += ["--save_pid", f"{os.getpid()}"]
        if args.enable_each_rank_log:
            deepspeed_launch.append(f"--enable_each_rank_log={args.enable_each_rank_log}")
        if args.elastic_training:
            deepspeed_launch.append("--enable_elastic_training")
            deepspeed_launch.append(f"--max_elastic_nodes={args.max_elastic_nodes}")
            deepspeed_launch.append(f"--min_elastic_nodes={args.min_elastic_nodes}")
        if args.bind_cores_to_rank:
            deepspeed_launch.append("--bind_cores_to_rank")
        if args.bind_core_list is not None:
            deepspeed_launch.append(f"--bind_core_list={args.bind_core_list}")
        cmd = deepspeed_launch + [args.user_script] + args.user_args # 在此处将 deepspeed_launch 函数 和 用户脚本 及 用户参数 拼接起来
    else:
        args.launcher = args.launcher.lower()
        if args.launcher == PDSH_LAUNCHER:
            runner = PDSHRunner(args, world_info_base64)
        elif args.launcher == OPENMPI_LAUNCHER:
            runner = OpenMPIRunner(args, world_info_base64, resource_pool)
        elif args.launcher == MPICH_LAUNCHER:
            runner = MPICHRunner(args, world_info_base64, resource_pool)
        elif args.launcher == IMPI_LAUNCHER:
            runner = IMPIRunner(args, world_info_base64, resource_pool)
        elif args.launcher == MVAPICH_LAUNCHER:
            runner = MVAPICHRunner(args, world_info_base64, resource_pool)
        elif args.launcher == SLURM_LAUNCHER:
            runner = SlurmRunner(args, world_info_base64, resource_pool)
        else:
            raise NotImplementedError(f"Unknown launcher {args.launcher}")

        if not runner.backend_exists():
            raise RuntimeError(f"launcher '{args.launcher}' not installed.")

        curr_path = os.path.abspath('.')
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = curr_path + ":" + env['PYTHONPATH']
        else:
            env['PYTHONPATH'] = curr_path

        excluded_vars = []
        for exclude_key, var_list in EXCLUDE_ENVS.items():
            if exclude_key in env.keys():
                # key exists in launcher env -> var list should be used
                excluded_vars += var_list

        # load envs from accelerator
        exports = EXPORT_ENVS + get_accelerator().export_envs()
        for var in env.keys():
            if any([var.startswith(name) for name in exports]):
                if not any([var == name for name in excluded_vars]):
                    runner.add_export(var, env[var])

        for environ_path in DEEPSPEED_ENVIRONMENT_PATHS:
            environ_file = os.path.join(environ_path, DEEPSPEED_ENVIRONMENT_NAME)
            if os.path.isfile(environ_file):
                logger.info(f"deepspeed_env file = {environ_file}")
                with open(environ_file, 'r') as fd:
                    for var in fd.readlines():
                        key, val = var.split('=', maxsplit=1)
                        runner.add_export(key, val)

        if args.launcher == PDSH_LAUNCHER:
            cmd, kill_cmd, env = runner.get_cmd(env, active_resources)
        else:
            cmd = runner.get_cmd(env, active_resources)

# cmd 内容：
# ['/home/mtn/miniconda3/envs/pytorch2.0/bin/python', '-u', '-m', 'deepspeed.launcher.launch', '--world_info=eyJsb2NhbGhvc3QiOiBbMCwgMV19',
# '--master_addr=127.0.0.1', '--master_port=29500', '--enable_each_rank_log=None', '--bind_cores_to_rank', 'cifar10_deepspeed.py', '--deepspeed']

result = subprocess.Popen(cmd, env=env) # 在此行启动子程序执行指令，启动子进程来执行我们的程序
```

**多进程调试**
```python
import os, debugpy
base_port = 5001
rank = int(os.getenv("LOCAL_RANK"))
debugpy.listen(("0.0.0.0", base_port + rank))
print("Waiting for debugger to attach...", os.getpid())
debugpy.wait_for_client()

debugpy.set_breakpoint()

# 在 vscode 中打开项目目录，点击运行和调试，添加如下 launch.json，再点击开始调试即可：（安装了c++ 扩展才行）
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
  {
      "name": "Python: Attach 5001",
      "type": "python",
      "request": "attach",
      "connect": {
          "host": "0.0.0.0",
          "port": 5001
      },
      "justMyCode": false
  },
  {
      "name": "Python: Attach 5002",
      "type": "python",
      "request": "attach",
      "connect": {
          "host": "0.0.0.0",
          "port": 5002
      },
      "justMyCode": true
  },
  ]
}
```

**deepspeed.launcher.launch 内容**
```python
def main():
    args = parse_args()

    import os, debugpy
    base_port = 29500
    rank = args.node_rank
    debugpy.listen(("127.0.0.1", base_port + rank))
    print("Waiting for debugger to attach...", os.getpid())
    debugpy.wait_for_client()
    debugpy.breakpoint()

    current_env = os.environ.copy()

    for k in current_env.keys():
        if "NCCL" in k:
            logger.info(f"{args.node_rank} {k}={current_env[k]}")

    if args.world_info == "None":
        raise ValueError("world_info can not be None")
    world_info = base64.urlsafe_b64decode(args.world_info)
    world_info = json.loads(world_info)

    logger.info(f"WORLD INFO DICT: {world_info}")
    node_list = list(world_info.keys())
    args.nnodes = len(node_list)
    local_node = node_list[args.node_rank]
    local_gpu_ids = world_info[local_node]
    num_local_procs = len(local_gpu_ids)
    logger.info(f"nnodes={args.nnodes}, num_local_procs={num_local_procs}, node_rank={args.node_rank}")

    global_rank_mapping = defaultdict(list)
    curr_global_rank = 0
    dist_world_size = 0
    for node_id in node_list:
        gids = world_info[node_id]
        dist_world_size += len(gids)
        for gid in gids:
            global_rank_mapping[node_id].append(curr_global_rank)
            curr_global_rank += 1
    logger.info(f"global_rank_mapping={global_rank_mapping}")
    logger.info(f"dist_world_size={dist_world_size}")
    current_env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, local_gpu_ids))
    logger.info(f"Setting CUDA_VISIBLE_DEVICES={current_env['CUDA_VISIBLE_DEVICES']}")

    # set PyTorch distributed related environmental variables
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["CROSS_RANK"] = str(args.node_rank)
    current_env["CROSS_SIZE"] = str(args.nnodes)
    current_env["LOCAL_SIZE"] = str(num_local_procs)

    if args.save_pid:
        print(f"launcher pid: {os.getpid()}")

    pid_file = None
    if args.save_pid:
        launcher_pid = os.getpid()
        pid_file = os.path.join(PID_FILE_BASEPATH, f"{args.save_pid}.deepspeed")
        assert not os.path.isfile(pid_file), "pid file exists but shouldn't"
        with open(pid_file, 'w') as fd:
            fd.write(f"{launcher_pid}")

    if not is_torch_elastic_compatible():
        if args.enable_elastic_training:
            logger.info(f"Disabling elastic training support as \
                    PyTorch version should be greater than 1.11.x")
            args.enable_elastic_training = False

    if os.path.exists(DLTS_POD_ENV_PATH):
        with open(DLTS_POD_ENV_PATH) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
            for line in lines:
                if line.startswith('export FC_TASKROLE_NAME') or line.startswith('export FC_TASK_INDEX'):
                    key_val = line.split()[1]
                    key, val = key_val.split('=')
                    current_env[key] = val

    processes = []
    cmd = []

    if not args.enable_elastic_training:
        if args.enable_each_rank_log != "None":
            # prepare the log path and the file name prefix
            if os.path.isfile(args.enable_each_rank_log):
                raise ValueError(f"{args.enable_each_rank_log} should not be a file, it should be a directory.")
            if not os.path.exists(args.enable_each_rank_log):
                try:
                    os.makedirs(args.enable_each_rank_log)
                except Exception as e:
                    print(e)
                    raise ValueError(f"unable to create directory {args.enable_each_rank_log} for each rank log.")
            log_name_prefix = time.strftime("%Y%m%d%H%M%S", time.localtime())

        for local_proc in range(0, num_local_procs):
            # each process's rank
            dist_rank = global_rank_mapping[local_node][local_proc]
            local_rank = dist_rank % num_local_procs
            current_env["RANK"] = str(dist_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            # spawn the processes
            cmd = []
            if args.bind_cores_to_rank:
                cores_per_rank, numactl_cmd = get_numactl_cmd(args.bind_core_list, num_local_procs, local_rank)
                current_env["OMP_NUM_THREADS"] = f"{cores_per_rank}"
                cmd = cmd + numactl_cmd
            if not args.no_python:
                cmd.append(sys.executable)
                cmd.append("-u")
                if args.module:
                    cmd.append("-m")
            else:
                if args.module:
                    raise ValueError("Don't use both the '--no_python' flag"
                                     " and the '--module' flag at the same time.")
            cmd.append(args.training_script)
            # A user may not want to pass local_rank as a keyword arg so we make this optional.
            if not args.no_local_rank:
                cmd.append(f"--local_rank={local_rank}")
            cmd += args.training_script_args

            if args.enable_each_rank_log != "None":
                log_file = os.path.join(args.enable_each_rank_log, f"{log_name_prefix}_rank{dist_rank}.log")
                log_fd = open(log_file, 'w')
                process = subprocess.Popen(cmd, env=current_env, stdout=log_fd, stderr=log_fd)
            else:
                process = subprocess.Popen(cmd, env=current_env)
            # logs the command from processes
            logger.info(f"process {process.pid} spawned with command: {cmd}")
            processes.append(process) # 添加到进程中
    else:
        from ..elasticity import DSElasticAgent
        from torch.distributed.elastic.rendezvous import RendezvousParameters
        from torch.distributed.elastic.agent.server.api import WorkerSpec
        import torch.distributed.elastic.rendezvous.registry as rdzv_registry
        from torch.distributed.elastic.multiprocessing import Std

        if args.min_elastic_nodes == -1:
            args.min_elastic_nodes = 1
        if args.max_elastic_nodes == -1:
            args.max_elastic_nodes = args.nnodes
        assert args.max_elastic_nodes > 0 and args.min_elastic_nodes > 0, "Max and Min nodes should be positive"

        current_env["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

        # Get config and arguments
        cmd = []
        if not args.no_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        else:
            if args.module:
                raise ValueError("Don't use both the '--no_python' flag"
                                 " and the '--module' flag at the same time.")
        cmd.append(args.training_script)
        cmd += args.training_script_args
        cmd_args = cmd[1:]

        rdzv_configs: Dict[str, str] = {'timeout': 100}
        run_id = os.environ.get("ELASTIC_RUN_ID", ELASTIC_TRAINING_ID_DEFAULT)

        # Creating config for rendezvous class
        rdzv_parameters = RendezvousParameters(backend='c10d',
                                               endpoint=args.master_addr + ":" + str(args.master_port),
                                               run_id=run_id,
                                               min_nodes=args.min_elastic_nodes,
                                               max_nodes=args.max_elastic_nodes,
                                               **rdzv_configs)

        spec = WorkerSpec(
            role='trainer',
            local_world_size=num_local_procs,
            entrypoint=cmd[0],
            args=cmd[1:],
            rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
            max_restarts=100,
            monitor_interval=5,
            redirects=Std.from_str("0"),
            tee=Std.from_str("0"),
            master_addr=None,
            master_port=None,
        )
        agent = DSElasticAgent(spec, current_env)
        agent.run()

    sig_names = {2: "SIGINT", 15: "SIGTERM"}
    last_return_code = None

    def sigkill_handler(signum, frame):
        for process in processes:
            logger.info(f"Killing subprocess {process.pid}")
            try:
                terminate_process_tree(process.pid)
            except Exception:
                pass
        if last_return_code is not None:
            logger.error(f"{cmd} exits with return code = {last_return_code}")
            sys.exit(last_return_code)
        if signum in sig_names:
            logger.info(f"Main process received {sig_names[signum]}, exiting")
        if args.save_pid:
            if os.path.isfile(pid_file):
                os.remove(pid_file)
        sys.exit(1)

    # pass SIGINT/SIGTERM to children if the parent is being terminated
    signal.signal(signal.SIGINT, sigkill_handler)
    signal.signal(signal.SIGTERM, sigkill_handler)

    alive_processes = set(processes)
    while len(alive_processes):
        finished_processes = []
        for process in alive_processes:
            if process.poll() is None:
                # the process is still running
                continue
            else:
                if process.returncode != 0:
                    last_return_code = process.returncode  # for sigkill_handler
                    sigkill_handler(signal.SIGTERM, None)  # not coming back
                else:
                    # exited cleanly
                    logger.info(f"Process {process.pid} exits successfully.")
                    finished_processes.append(process)
        alive_processes = set(alive_processes) - set(finished_processes)

        time.sleep(1)
```

**cuda accelerate 类来监控device测配置**
```python
# /home/mtn/DeepSpeed/deepspeed/accelerator/cuda_accelerator.py
class CUDA_Accelerator(DeepSpeedAccelerator):

    def __init__(self):
        self._name = 'cuda'
        self._communication_backend_name = 'nccl'
        if pynvml is None:
            self._init_pynvml()

    def _init_pynvml(self):
        global pynvml
        try:
            import pynvml
        except ImportError:
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            pynvml = None
            return

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'cuda'
        return 'cuda:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.cuda.device(device_index)

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def current_device(self):
        return torch.cuda.current_device()

    def current_device_name(self):
        return 'cuda:{}'.format(torch.cuda.current_device())

    def device_count(self):
        return torch.cuda.device_count()

    def synchronize(self, device_index=None):
        return torch.cuda.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def get_rng_state(self, device_index=None):
        if device_index is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device_index)

    def manual_seed(self, seed):
        return torch.cuda.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch.cuda.manual_seed_all(seed)

    def initial_seed(self, seed):
        return torch.cuda.initial_seed(seed)

    def default_generator(self, device_index):
        return torch.cuda.default_generators[device_index]

    ...
```

# 3 训练脚本解读

## 3.1 单进程读取数据
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pytorch在分布式训练过程中，对于数据的读取是采用主进程预读取并缓存，然后其它进程从缓存中读取，不同进程之间的数据同步具体通过torch.distributed.barrier()实现, torch中采用了barrier()函数对其它非主进程进行阻塞，来达到同步的目的。<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;torch.distributed.barrier()，设置一个阻塞栅栏，让符合条件进程处于等待状态，等待所有进程到达栅栏处（包括主进程数据处理完毕）；如果处理 create dataloader()函数的进程是主进程，其会直接去读取数据并处理，然后其处理结束之后会接着遇到torch.distributed.barrier()，此时，所有进程都到达了当前的栅栏处，这样所有进程就达到了同步，并同时得到释放。<br>

**要点是 不同进程可能在不同位置到达 barrier 处** <br>


## 3.2 ds 相关配置信息
```python
{'train_batch_size': 16, 'steps_per_print': 2000, 'optimizer': {'type': 'Adam', 'params': {...}}, 'scheduler': {'type': 'WarmupLR', 'params': {...}}, 
'gradient_clipping': 1.0, 'prescale_gradients': False, 'bf16': {'enabled': False}, 'fp16': {'enabled': True, 'fp16_master_weights_and_grads': False, 'loss_scale': 0, 
'loss_scale_window': 500, 'hysteresis': 2, 'min_loss_scale': 1, 'initial_scale_power': 15}, 'wall_clock_breakdown': False, 
'zero_optimization': {'stage': 0, 'allgather_partitions': True, 'reduce_scatter': True, 'allgather_bucket_size': 50000000, 
'reduce_bucket_size': 50000000, 'overlap_comm': True, 'contiguous_gradients': True, 'cpu_offload': False}}
```

## 3.3 deepspeed.initialize

- /home/mtn/DeepSpeed/deepspeed/__init__.py

### 3.3.1 用户接口** <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;先初始化DeepSpeed配置(即DeepSpeedConfig)，然后初始化DeepSpeed引擎(DeepSpeedHybridEngine/DeepSpeedEngine/PipelineEngint). <br>

　　返回engine、engine的optimizer、engine的training_dataloader、engine的lr_scheduler；
```python
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        training_data=trainset,
        config=ds_config,
    )
```

### 3.3.2 initialize 中通信后端初始化
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;始化加速器并更新到全局变量ds_accelerator(详见get_accelerator函数)，并初始化分布式计算后端框架(用于多个计算节点协同工作以加速训练，处理模型参数和梯度同步、通信等操作，存入全局变量ccl_backend，CCLBackend类，详见init_deepspeed_backend函数)；<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;调用父类TorchBackend完成初始化(实际上调用torch.distributed.init_process_group初始化torch分布式环环境中进程组)；<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过ccl_comm_op(即CCLCommBuilder)的get_kvs_addr创建一个主要的key-value存储器并将其内存地址广播(TorchBackend的broadcast方法)给其他机器, 根据当前机器编号(rank)、key-value存储器地址等初始化ccl_comm_op(即CCLCommBuilder类) <br>

- available_coll：通信操作名列表(broadcast、all_reduce、inference_all_reduce、all_reduce_caching、barrier)，详见ccl_comm_op的get_available_coll；<br>

```python
# cdb = TorchBackend(dist_backend, timeout, init_method, rank, world_size)
# TorchBackend 中对 pytorch 中的分布式 进行了封装
from deepspeed import comm as dist
dist_backend = get_accelerator().communication_backend_name()
dist.init_distributed(dist_backend=dist_backend, dist_init_required=dist_init_required)
```

### 3.3.3 DeepSpeedConfig 初始化
```python
# DeepSpeedConfig初始化
# _param_dict：传入的配置dict；
# 根据_param_dict初始化如下变量，详见_initialize_params函数
train_batch_size：
train_micro_batch_size_per_gpu：
gradient_accumulation_steps：
steps_per_print：配置中取值10；
prescale_gradients：配置中配置为false；
gradient_clipping：配置中取值为1.0；
zero_config：zero_optimization配置参数；
zero_optimization_stage：zero_optimization配置参数中"stage"参数；
zero_enabled：zero_optimization_stage是否大于0；
……
activation_checkpointing_config：初始化DeepSpeedActivationCheckpointingConfig类；
comms_config：初始化DeepSpeedCommsConfig类；
flops_profiler_config：初始化DeepSpeedFlopsProfilerConfig类；
autotuning_config：初始化DeepSpeedAutotuningConfig类；
nebula_config：初始化DeepSpeedNebulaConfig类；
weight_quantization_config：初始化WeightQuantConfig类；
 设置训练batch_size相关参数(如果没有配置则根据其他配置计算)，详见_configure_train_batch_size函数
gradient_accumulation_steps：train_batch_size/(train_micro_batch_size_per_gpu*分布式进程个数)；
train_micro_batch_size_per_gpu：train_batch_size/(gradient_accumulation_steps*分布式进程个数)
train_batch_size：train_micro_batch_size_per_gpu*分布式进程个数*gradient_accumulation_steps
train_micro_batch_size_per_gpu：train_batch_size/分布式进程个数(如果没有配置gradient_accumulation_steps，则gradient_accumulation_steps设置为1)；
gradient_accumulation_steps：train_micro_batch_size_per_gpu*分布式进程个数(如果没有配置gradient_accumulation_steps，则gradient_accumulation_steps设置为1)；
 合法性检测，详见_do_sanity_check函数；
```

### 3.3.4 


# 4 参考文献
- [参考文档](https://www.zhangzhenhu.com/deepspeed/stage2-%E5%88%9D%E5%A7%8B%E5%8C%96.html)




