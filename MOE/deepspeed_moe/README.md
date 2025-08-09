# 1. E + D parallel
- world_size = 16
- expert_parallel_size = 2 # number of experts in same group
- data_parallel_group = [0,1,...,15] - all reduce is only on non-MoE
- expert_data_parallel_group = [0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15] - all reduce is only on MoE params
- expert_parallel_group = [0,1], [2,3], [4,5], [6,7], [8,9] - no all reduce, but all to all
- use_data_before_expert_parallel_ (bool): Use the D + E instead of E + D topology


# 2 E + M + D parallel
- world_size = 16
- model_degree = 2
- expert_degree = 4 # number of experts in same group
- data_parallel_group =[0,2,4,6,8,10,12,14],                 [1,3,5,7,9,11,13,15]
- mp_group = [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]
- expert_parallel_group = [0,2,4,6], [8,10,12,14]             [1,3,5,7], [9,11,13,15]
- expert_data_parallel_group = [0,8],[2,10],[4,12],[6,14],    [1,9],[3,11],[5,13],[7,15]



