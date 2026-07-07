# 0 CUDA Graph 需要的数据结构

```c++
  struct ACEGraphInfo {
    CUgraph graph;
    CUgraphExec graphExec;
    std::vector<std::vector<CUgraphNode>> nodes; // 每个 peer 需要多个node
    CUcontext ctx;

    std::vector<CUDA_MEM_TRANSFER_NODE_PARAMS> copyParams;
    std::vector<CUDA_MEM_ATOMIC_NODE_PARAMS> atomicParams;
    std::vector<CUDA_MEM_WAIT_WRITE_NODE_PARAMS> waitParams;

    uint64_t seqNum{0};
  };
```

- 几个重要概念
```sh
可以分清几个概念：
CUcontext：设备上下文，决定这些 driver-level 资源属于哪个 device/context。
CUgraph：graph 的静态 DAG 描述。
CUgraphExec：实例化后可 launch 的 graph。
CUstream / cudaStream_t：实际 launch 时排队执行的 stream。
```

- ctx 不是 stream，也不是同步工具。它更像 graph node 的“设备环境”。当前代码里先在 init 阶段保存 ctx，之后每轮 collective 更新 node 参数时继续用同一个 ctx，保证这些 mem atomic / wait / transfer node 都在创建 graph 时对应的设备上下文下操作。
- 如果 context 不对，常见问题就是 graph node 里的设备地址、peer 地址、mem op 参数可能无法在当前 context 下合法解释，driver API 可能报 invalid context / invalid value，或者行为不符合预期。

# 1. graph mode 执行流程

```c++
// step 1 : 初始化 graph and ctx
CUgraph graph;
CUgraphExec graphExec;
CUcontext ctx;
C10_CUDA_DRIVER_CHECK(cuCtxGetCurrent(&ctx));
C10_CUDA_DRIVER_CHECK(cuGraphCreate(&(graph), 0));

// step 2: node add to graph : 主要这里的会建立节点间的依赖关系
CUresult CUDAAPI cuGraphAddMemWaitWriteNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_WAIT_WRITE_NODE_PARAMS* memWaitWriteParams, CUcontext ctx);
CUresult CUDAAPI cuGraphAddMemTransferNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_TRANSFER_NODE_PARAMS* memTransferParams, CUcontext ctx);
CUresult CUDAAPI cuGraphAddMemAtomicValueNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS* memAtomicValueParams, CUcontext ctx);
CUresult CUDAAPI cuGraphAddMemAtomicNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_ATOMIC_NODE_PARAMS* memAtomicParams, CUcontext ctx);

// step 3: instantiate graph
C10_CUDA_DRIVER_CHECK(cuGraphInstantiate(graphExec, graph, 1));

// step 4: update node
C10_CUDA_DRIVER_CHECK(cuGraphExecMemAtomicNodeSetParams(graphExec, CUgraphNode, NODE_PARAMS, ctx));
C10_CUDA_DRIVER_CHECK(cuGraphExecMemWaitWriteNodeSetParams(graphExec, CUgraphNode, NODE_PARAMS, ctx));
C10_CUDA_DRIVER_CHECK(cuGraphExecMemAtomicNodeSetParams(graphExec, CUgraphNode, NODE_PARAMS, ctx));

// step 5: launch graph
C10_CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
```

# 2. 添加不同节点到 graph 中

## 2.1 CUDA_MEM_ATOMIC_NODE_PARAMS

- 对 elementCount 个元素执行 dst[i] atomic_op src[i]

```c++
typedef enum CUatomicType_enum {
    CU_ATOMIC_TYPE_ATOMIC_ADD32 = 0,
    CU_ATOMIC_TYPE_ATOMIC_ADD64,
    CU_ATOMIC_TYPE_ATOMIC_UMIN32,
    CU_ATOMIC_TYPE_ATOMIC_UMIN64,
    CU_ATOMIC_TYPE_ATOMIC_IMIN32,
    CU_ATOMIC_TYPE_ATOMIC_IMIN64,
    CU_ATOMIC_TYPE_ATOMIC_UMAX32,
    CU_ATOMIC_TYPE_ATOMIC_UMAX64,
    CU_ATOMIC_TYPE_ATOMIC_IMAX32,
    CU_ATOMIC_TYPE_ATOMIC_IMAX64,
    CU_ATOMIC_TYPE_ATOMIC_ADD_F32,
    CU_ATOMIC_TYPE_ATOMIC_ADD_F64,
    CU_ATOMIC_TYPE_ATOMIC_ADD_HF16,
    CU_ATOMIC_TYPE_ATOMIC_ADD_BF16,
} CUatomicType;

typedef struct CUDA_MEM_ATOMIC_NODE_PARAMS_st {
    CUdeviceptr dst;
    CUdeviceptr src;
    size_t elementCount;
    CUatomicType operation;
} CUDA_MEM_ATOMIC_NODE_PARAMS_v1;
typedef CUDA_MEM_ATOMIC_NODE_PARAMS_v1 CUDA_MEM_ATOMIC_NODE_PARAMS;

/**
 * \brief Creates a memory atomic node and adds it to a graph
 *
 * Creates a new memory atomic node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will perform the memory atomic operation described by \p memAtomicParams.
 * See ::cuMemoryAtomicAsync() for a description of the structure.

 * \param phGraphNode     - Returns newly created node
 * \param hGraph          - Graph to which to add the node
 * \param dependencies    - Dependencies of the node
 * \param numDependencies - Number of dependencies
 * \param memAtomicParams - Parameters for the memory atomic
 * \param ctx             - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 * \sa
 */
CUresult CUDAAPI cuGraphAddMemAtomicNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_ATOMIC_NODE_PARAMS* memAtomicParams, CUcontext ctx);
```

## 2.2. CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS

- 对 *dst 执行 atomic_op value;

```c++
typedef enum CUatomicValueType_enum {
    CU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64 = 0,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_SUB64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_UMIN64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_IMIN64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_UMAX64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_IMAX64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_AND64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_OR64,
    CU_ATOMIC_VALUE_TYPE_ATOMIC_XOR64,
} CUatomicValueType;

typedef struct CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS_st {
    CUdeviceptr dst;
    cuuint64_t value;
    CUatomicValueType operation;
} CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS_v1;
typedef CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS_v1 CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS;
```

- 对该node的入图: <br>

```c++
/**
 * \brief Creates a memory atomic value node and adds it to a graph
 *
 * Creates a new memory atomic value node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will perform the memory atomic value operation described by \p memAtomicValueParams.
 * See ::cuMemoryAtomicValueAsync() for a description of the structure.

 * \param phGraphNode          - Returns newly created node
 * \param hGraph               - Graph to which to add the node
 * \param dependencies         - Dependencies of the node
 * \param numDependencies      - Number of dependencies
 * \param memAtomicValueParams - Parameters for the memory atomic value
 * \param ctx                  - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 */
CUresult CUDAAPI cuGraphAddMemAtomicValueNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_ATOMIC_VALUE_NODE_PARAMS* memAtomicValueParams, CUcontext ctx);
```

## 2.3 CUDA_MEM_WAIT_WRITE_NODE_PARAMS

- 描述一个 Graph 节点要执行的 memory wait / memory write 操作。

```c++
/**
 * Operations for ::cuStreamBatchMemOp
 */
typedef enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32  = 1,     /**< Represents a ::cuStreamWaitValue32 operation */
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,     /**< Represents a ::cuStreamWriteValue32 operation */
    CU_STREAM_MEM_OP_WAIT_VALUE_64  = 4,     /**< Represents a ::cuStreamWaitValue64 operation */
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,     /**< Represents a ::cuStreamWriteValue64 operation */
    CU_STREAM_MEM_OP_BARRIER = 6,            /**< Insert a memory barrier of the specified type */
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3 /**< This has the same effect as ::CU_STREAM_WAIT_VALUE_FLUSH, but as a
                                                  standalone operation. */
} CUstreamBatchMemOpType;

typedef struct CUDA_MEM_WAIT_WRITE_NODE_PARAMS_st {
    CUdeviceptr addr;
    cuuint64_t value;
    unsigned int flags;
    CUstreamBatchMemOpType operation;
} CUDA_MEM_WAIT_WRITE_NODE_PARAMS_v1;
typedef CUDA_MEM_WAIT_WRITE_NODE_PARAMS_v1 CUDA_MEM_WAIT_WRITE_NODE_PARAMS;
```

- **添加节点到graph** <br>

```c++
/**
 * \brief Creates a memory wait / write value node and adds it to a graph
 *
 * Creates a new memory wait / write value node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will perform the memory wait / write value described by \p memWaitWriteParams.

 * \param phGraphNode        - Returns newly created node
 * \param hGraph             - Graph to which to add the node
 * \param dependencies       - Dependencies of the node
 * \param numDependencies    - Number of dependencies
 * \param memWaitWriteParams - Parameters for the memory atomic value
 * \param ctx                - Context on which to run the node
 *
 *
 * The CUDA_MEM_WAIT_WRITE_NODE_PARAMS structure is defined as:
 *
 * \code
 *  typedef struct CUDA_MEM_WAIT_WRITE_NODE_PARAMS_st {
 *      CUdeviceptr addr;
 *      cuuint64_t value;
 *      unsigned int flags;
 *      CUstreamBatchMemOpType operation;
 *  } CUDA_MEM_WAIT_WRITE_NODE_PARAMS;
 * \endcode
 *
 * See ::CUstreamBatchMemOpType for the full set of supported operations, and
 * ::cuStreamWaitValue32(), ::cuStreamWaitValue64(), ::cuStreamWriteValue32(),
 * and ::cuStreamWriteValue64() for details of specific operations.
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 */
CUresult CUDAAPI cuGraphAddMemWaitWriteNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_WAIT_WRITE_NODE_PARAMS* memWaitWriteParams, CUcontext ctx);
```

## 2.4. CUDA_MEM_TRANSFER_NODE_PARAMS

- 定义在 CUDA driver 头文件: /usr/local/CUDA/include/CUDA.h:5116;
- 作用: 它是 CUDA graph 里 memory transfer node 的参数结构体，描述一次设备内存拷贝/传输;
- 往 CUDA Graph 里添加一个 memory transfer 节点。Graph launch 执行到这个节点时，会按参数做一段内存搬运.

```c++
typedef struct CUDA_MEM_TRANSFER_NODE_PARAMS_st {
    CUdeviceptr dst;  // 目标地址
    CUdeviceptr src;  // 源地址
    size_t ByteCount; // copy 字节数
} CUDA_MEM_TRANSFER_NODE_PARAMS_v1;
typedef CUDA_MEM_TRANSFER_NODE_PARAMS_v1 CUDA_MEM_TRANSFER_NODE_PARAMS;

// 多个params 组成 vector 完成整个 params 的数据copy
std::vector<CUDA_MEM_TRANSFER_NODE_PARAMS> copyParams;

/**
 * \brief Creates a memory transfer node and adds it to a graph
 *
 * Creates a new memory transfer node and adds it to \p hGraph with \p numDependencies
 * dependencies specified via \p dependencies.
 * It is possible for \p numDependencies to be 0, in which case the node will be placed
 * at the root of the graph. \p dependencies may not have any duplicate entries.
 * A handle to the new node will be returned in \p phGraphNode.
 *
 * When the graph is launched, the node will perform the memory transfer described by \p memTransferParams.
 * See ::cuMemoryTransfer() for a description of the structure and its restrictions.
 *
 * \param phGraphNode       - Returns newly created node
 * \param hGraph            - Graph to which to add the node
 * \param dependencies      - Dependencies of the node
 * \param numDependencies   - Number of dependencies
 * \param memTransferParams - Parameters for the memory transfer
 * \param ctx               - Context on which to run the node
 *
 * \return
 * ::CUDA_SUCCESS,
 * ::CUDA_ERROR_DEINITIALIZED,
 * ::CUDA_ERROR_NOT_INITIALIZED,
 * ::CUDA_ERROR_INVALID_VALUE
 * \note_graph_thread_safety
 * \notefnerr
 *
 */
CUresult CUDAAPI cuGraphAddMemTransferNode(CUgraphNode *phGraphNode, CUgraph hGraph, const CUgraphNode *dependencies, size_t numDependencies, const CUDA_MEM_TRANSFER_NODE_PARAMS* memTransferParams, CUcontext ctx);
```
