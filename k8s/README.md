下面这些是真实的 k9s 界面截图参考，不同版本和主题会略有差异，但整体结构一致：顶部是集群信息，中间是资源列表，右上/底部是快捷键提示。

[![Accelerate your Kubernetes Administration with k9s](https://images.openai.com/static-rsc-4/8-EJz6gK2huisnwk9jYMIvbVsZp6aZtHQ0GDKpUIPTy0AbOcF_e5HfJoZ6pV3xMaPNVON6esoq7RvRdfTWcy_uauFujV8dKpoREe4Esvh4xP5R0vN-FeDFNbnbm0Ni7Bl6w4nbqFQiCWMq0hHnLeNMqOyagW_6ZCA8nZ4f1CqAI?purpose=inline)](https://www.dae.mn/blog/accelerate-your-kubernetes-administration-with-k9s?utm_source=chatgpt.com)

# k9s 入门教程：理解 Node、Namespace、Pod，并用 k9s 排查问题

## 1. k9s 是什么？

`k9s` 是一个 Kubernetes 的终端 UI 工具，可以理解为：

```bash
kubectl 的交互式图形终端版
```

它会连接当前机器上的 Kubernetes 配置，也就是通常的：

```bash
~/.kube/config
```

然后以终端界面的方式展示 Kubernetes 资源，例如：

```text
Pod
Node
Namespace
Deployment
Service
ConfigMap
Secret
Event
Log
```

官方对 k9s 的定位是：一个用于和 Kubernetes 集群交互的 terminal UI，可以持续 watch 集群变化，并提供快捷操作。([GitHub][1])

---

## 2. Kubernetes 里的核心关系

先记住这几个概念：

| 概念         | 含义               | 类比          |
| ---------- | ---------------- | ----------- |
| Cluster    | 整个 Kubernetes 集群 | 一个机房 / 一套平台 |
| Node       | 真正运行容器的机器        | 一台服务器       |
| Namespace  | 逻辑隔离空间           | 项目组 / 环境    |
| Pod        | K8s 最小运行单元       | 一个应用实例      |
| Deployment | 管理 Pod 副本        | 应用部署配置      |
| Service    | 给 Pod 提供访问入口     | 服务入口 / 负载均衡 |

Kubernetes 官方文档里也说明，namespace 是给 Pod、Service、Deployment 等资源提供作用域的逻辑空间。([Kubernetes][2])

---

## 3. 为什么一个 worker 上会看到很多 namespace？

你看到的现象一般是：

```text
某个 worker node 上有很多 namespace
```

更准确地说应该是：

```text
某个 worker node 上运行了来自多个 namespace 的 Pod
```

因为：

```text
namespace 不是运行在 worker 上的实体
node 才是真实机器
pod 才是真正被调度到 node 上运行的对象
```

关系图如下：

```text
Kubernetes Cluster
│
├── Namespace: default
│   └── Pod: nginx-xxx
│
├── Namespace: dev
│   └── Pod: backend-xxx
│
├── Namespace: prod
│   └── Pod: api-xxx
│
└── Namespace: kube-system
    └── Pod: kube-proxy-xxx


Worker Node: worker-1
│
├── default/nginx-xxx
├── dev/backend-xxx
├── prod/api-xxx
└── kube-system/kube-proxy-xxx
```

所以你在 `worker-1` 上看到很多 namespace，本质是：

```text
worker-1 上运行了多个 namespace 下的 Pod
```

不是：

```text
worker-1 拥有多个 namespace
```

---

## 4. k9s 界面怎么看？

一个典型 k9s 界面大概长这样：

```text
Context: my-cluster
Cluster: my-cluster
User:    admin
K9s Rev: v0.xx.x
K8s Rev: v1.xx.x
CPU:     xx%
MEM:     xx%

Pods(all)[23]
┌──────────────────────────────────────────────────────────────┐
│ NAMESPACE     NAME              READY  STATUS   IP       NODE │
│ default       nginx-xxx         1/1    Running  10.x.x.x worker-1 │
│ dev           backend-xxx       1/1    Running  10.x.x.x worker-2 │
│ prod          api-xxx           1/1    Running  10.x.x.x worker-1 │
│ kube-system   kube-proxy-xxx    1/1    Running  10.x.x.x worker-1 │
└──────────────────────────────────────────────────────────────┘
```

可以分成三块：

```text
顶部：当前 context、cluster、user、版本、CPU/MEM
中间：当前资源列表，例如 Pods、Nodes、Deployments
右侧/底部：快捷键提示
```

---

## 5. k9s 常用资源页面

在 k9s 里按：

```text
:
```

可以输入资源名称，例如：

```text
:pods
:nodes
:ns
:deploy
:svc
:events
```

常用命令如下：

| k9s 输入             | 含义            | 对应 kubectl           |
| ------------------ | ------------- | -------------------- |
| `:pods` / `:po`    | 查看 Pod        | `kubectl get pods`   |
| `:nodes` / `:node` | 查看 Node       | `kubectl get nodes`  |
| `:ns`              | 查看 Namespace  | `kubectl get ns`     |
| `:deploy`          | 查看 Deployment | `kubectl get deploy` |
| `:svc`             | 查看 Service    | `kubectl get svc`    |
| `:events`          | 查看事件          | `kubectl get events` |

k9s 官方命令文档说明，可以通过 `:` 加资源名进入对应资源视图，例如 `:pod`，也支持 short-name 或 alias。([k9scli.io][3])

---

## 6. 如何查看所有 namespace 的 Pod？

进入 Pod 页面：

```text
:pods
```

然后按：

```text
0
```

`0` 表示切换到 **all namespaces**。

效果类似于：

```bash
kubectl get pods -A
```

一些 k9s 教程也明确说明，在 Pod 页面按 `0` 可以查看所有 namespace 下的 Pod。([KodeKloud Notes][4])

---

## 7. 如何查找某个 Node？

进入 Node 页面：

```text
:node
```

或者：

```text
:nodes
```

然后按 `/` 搜索，例如：

```text
/worker-1
```

如果要看 node 的详细信息，选中该 node 后按：

```text
d
```

相当于：

```bash
kubectl describe node worker-1
```

---

## 8. 如何查看某个 worker 上运行了哪些 Pod？

### 方法一：在 k9s 里查看

进入 Pod 页面：

```text
:pods
```

切换所有 namespace：

```text
0
```

然后搜索 node 名：

```text
/worker-1
```

如果 Pod 列表里有 `NODE` 字段，就能看到哪些 Pod 跑在这个 node 上。

---

### 方法二：用 kubectl 查看

```bash
kubectl get pods -A -o wide | grep worker-1
```

示例输出：

```text
NAMESPACE      NAME               READY   STATUS    NODE
default        nginx-xxx          1/1     Running   worker-1
dev            backend-xxx        1/1     Running   worker-1
prod           api-xxx            1/1     Running   worker-1
kube-system    kube-proxy-xxx     1/1     Running   worker-1
```

这说明：

```text
worker-1 上运行了 default、dev、prod、kube-system 等 namespace 的 Pod
```

---

## 9. k9s 常用快捷键

| 快捷键        | 作用                        |
| ---------- | ------------------------- |
| `:`        | 输入资源类型，例如 `:pods`、`:node` |
| `/`        | 搜索 / 过滤                   |
| `0`        | 查看所有 namespace            |
| `d`        | describe 当前资源             |
| `l`        | 查看 Pod 日志                 |
| `s`        | 进入 Pod shell              |
| `y`        | 查看 YAML                   |
| `e`        | 编辑资源                      |
| `Ctrl + d` | 删除资源                      |
| `q`        | 返回 / 退出                   |
| `Esc`      | 取消搜索或返回                   |
| `Enter`    | 进入当前资源详情                  |

k9s 官方文档中也列出了 `?` 查看快捷键、`ctrl-a` 查看资源别名、`:q` 或 `ctrl-c` 退出、`:pod` 查看资源等命令。([k9scli.io][3])

---

## 10. 常见排查流程

假设你要排查一个异常 Pod。

### 第一步：进入 Pod 页面

```text
:pods
```

### 第二步：查看所有 namespace

```text
0
```

### 第三步：搜索目标 Pod

```text
/api
```

或者：

```text
/backend
```

### 第四步：看状态

重点看这些字段：

```text
READY
STATUS
RESTARTS
NODE
AGE
```

常见状态含义：

| 状态                 | 含义          |
| ------------------ | ----------- |
| `Running`          | 正常运行        |
| `Pending`          | 还没调度成功      |
| `CrashLoopBackOff` | 容器反复崩溃      |
| `ImagePullBackOff` | 镜像拉取失败      |
| `Error`            | 容器异常退出      |
| `Completed`        | 任务型 Pod 已完成 |

---

## 11. 排查 Pod 的标准动作

选中目标 Pod 后：

### 查看日志

```text
l
```

类似于：

```bash
kubectl logs pod-name -n namespace
```

### 查看详细信息

```text
d
```

类似于：

```bash
kubectl describe pod pod-name -n namespace
```

重点看 `Events` 部分：

```text
Events:
  FailedScheduling
  FailedMount
  FailedPullImage
  BackOff
```

### 进入容器 shell

```text
s
```

类似于：

```bash
kubectl exec -it pod-name -n namespace -- sh
```

有些 k9s 教程也说明，选中 Pod 后可以用 `l` 查看日志、`d` 查看 describe、`s` 进入 shell。([DEV Community][5])

---

## 12. 如何理解 namespace 和 node 的关系？

可以用这张图记忆：

```text
Namespace 是逻辑空间
Node 是真实机器
Pod 属于某个 Namespace
Pod 被调度到某个 Node


Namespace: dev
└── Pod: dev/backend-1
    └── Node: worker-1


Namespace: prod
└── Pod: prod/api-1
    └── Node: worker-1


Namespace: kube-system
└── Pod: kube-system/kube-proxy
    └── Node: worker-1


最终 worker-1 上看到：

worker-1
├── dev/backend-1
├── prod/api-1
└── kube-system/kube-proxy
```

一句话总结：

```text
namespace 管逻辑隔离，node 管实际运行，pod 同时属于 namespace 并运行在某个 node 上。
```

---

## 13. 想让某个 namespace 的 Pod 固定跑到某些 node 上怎么办？

默认情况下，namespace 不会自动绑定 node。

如果想控制 Pod 调度位置，需要配置调度策略，例如：

```text
nodeSelector
nodeAffinity
taints / tolerations
```

例如给 node 打标签：

```bash
kubectl label node worker-1 env=dev
```

然后 Pod 配置：

```yaml
spec:
  nodeSelector:
    env: dev
```

这样这个 Pod 才会被调度到带有 `env=dev` 标签的 node 上。

---

## 14. 新手建议学习路径

按这个顺序熟悉 k9s：

```text
1. k9s        启动 k9s
2. :ns        查看 namespace
3. :node      查看 node
4. :pods      查看 pod
5. 0          查看所有 namespace
6. /xxx       搜索资源
7. d          describe 资源
8. l          查看日志
9. y          查看 YAML
10. :deploy   查看 deployment
11. :svc      查看 service
12. :events   查看事件
```

最常用组合：

```text
:pods
0
/关键词
d
l
```

含义是：

```text
进入 Pod 页面
查看所有 namespace
搜索目标 Pod
查看详细信息
查看日志
```

---

## 15. 最重要的总结

```text
k9s 是 Kubernetes 的终端管理界面。

namespace 是逻辑隔离空间，不是机器。

node 是实际运行容器的机器。

pod 属于某个 namespace，同时被调度到某个 node 上运行。

一个 worker 上看到很多 namespace，说明这个 worker 上运行了来自多个 namespace 的 Pod。

在 k9s 里：
:pods 查看 Pod
:node 查看 Node
:ns 查看 Namespace
0 查看所有 namespace
/ 搜索
d 查看 describe
l 查看日志
s 进入 shell
```

实际排查时，最常用的是：

```text
:pods → 0 → /关键词 → d → l
```

[1]: https://github.com/derailed/k9s?utm_source=chatgpt.com "derailed/k9s: 🐶 Kubernetes CLI To Manage Your Clusters ..."
[2]: https://kubernetes.io/docs/tutorials/cluster-management/namespaces-walkthrough/?utm_source=chatgpt.com "Namespaces Walkthrough"
[3]: https://k9scli.io/topics/commands/?utm_source=chatgpt.com "Commands"
[4]: https://notes.kodekloud.com/docs/Kubernetes-Troubleshooting-for-Application-Developers/Prerequisites/k9s-Walkthrough/page?utm_source=chatgpt.com "k9s Walkthrough"
[5]: https://dev.to/aws-builders/k9s-manage-your-kubernetes-cluster-like-a-pro-lko?utm_source=chatgpt.com "k9s - manage your Kubernetes cluster and it's objects like a ..."
