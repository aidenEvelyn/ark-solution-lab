# ark-solution-lab · 方舟售前方案实验室

> 客户一句话需求 → **带证据的技术方案草稿 + 追问清单 + 成本报告**；
> 并且用一个可复现的对照实验回答一个问题：**这个场景到底该不该上多智能体。**

**状态：🚧 施工中**（第 1/7 步完成：骨架与设计文档）

---

## 这个仓库是什么

企业客户找到大模型平台时，说的往往是这种话：

> 「我们想做个内部知识库问答，但数据不能出内网，QPS 大概 50，预算每月两千。」

售前要做的第一件事，是把它变成一份**能落地的技术方案**：选哪个模型、怎么调、有哪些限制、大概多少钱、哪些地方客户没说清楚需要追问。

这件事有三个真实的痛点：

1. **模型会编。** 方案里写「这个模型支持 128k 上下文」——是真的还是编的？售前把它发出去，就是在拿公司信誉背书。
2. **客户约束会漏。** 客户说了「数据不出内网」，方案里通篇没提私有化——这种漏项在 PoC 阶段才发现，代价很高。
3. **需求常常是欠规格的。** 该追问的时候硬着头皮出方案，后面全是返工。

这个项目把这三件事做成可测的环节，然后回答一个更根本的问题：

> **要解决它，到底需要几个 agent？多智能体比单 Agent 强在哪、贵在哪、什么情况下不值？**

---

## 它长什么样

```
                         客户一句话需求
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Intake · 需求理解    │  单线程
                    │  抽结构化需求 +       │  （信息不足时直接出追问清单，
                    │  列「未明确的点」      │    不硬编方案）
                    └──────────┬───────────┘
                               │  结构化 Requirement（过契约校验）
                               ▼
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │ Evidence #1 │      │ Evidence #2 │      │ Evidence #3 │   并行 · 只读
   │ 模型选型     │      │ 限制与合规   │      │ 成本与性能   │   各自检索知识库，
   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘   互不写入
          └────────────────────┼────────────────────┘
                               │  证据包（每条带来源）
                               ▼
                    ┌──────────────────────┐
                    │  Writer · 方案撰写    │  单线程
                    │  每条技术断言必须挂   │  （唯一允许「写入」的环节）
                    │  到一条证据上         │
                    └──────────┬───────────┘
                               │  方案草稿 + 引用表
                               ▼
                    ┌──────────────────────┐
                    │  Critic · 独立审校    │  逐条核对：
                    │  不合格 → 退回重写    │  ① 有没有无依据的断言
                    │  （最多 2 轮）        │  ② 客户约束有没有被回应
                    └──────────┬───────────┘
                               ▼
              方案草稿 + 证据表 + 追问清单 + 成本/延迟报告
```

**为什么这么拆**（依据在 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)，每条都带出处）：

- 只在**只读、可并行**的取证环节用多个 agent —— 这是目前有实证支持的多智能体适用区间
- **写入保持单线程** —— Cognition 的工程结论：并行 agent 各自携带的隐含决策会互相冲突，产出不一致的结果
- 审校**独立**于撰写 —— 自己检查自己，等于没检查

---

## 对照实验（本仓库的重点）

同一批测试用例，跑三条流水线，用同一套指标打分：

| 流水线 | 是什么 | 想回答的问题 |
|---|---|---|
| **A · 单 Agent 一把梭** | 一个 prompt，直接出方案 | 最省事的做法差在哪 |
| **B · 单 Agent + 结构化自检** | 强制 JSON 输出 + 让它自查一遍 | **便宜的改良能不能顶替多智能体** |
| **C · 多智能体（上图）** | 并行取证 + 单线程写 + 独立审校 | 多花的钱买到了什么 |

指标：无依据断言率 · 客户约束覆盖率 · 契约合规率 · 该追问时是否追问 · 单次成本（元）· 单次延迟。

> ⚠️ **诚实声明**：测试用例是自建的、样本量小（约 20 条），结论只在**这个设定下**成立，不是行业基准。
> 详细的口径、打假条件与局限见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

---

## 快速开始

```bash
git clone <this-repo>
cd ark-solution-lab

# 零依赖：核心管线只用 Python 标准库
python -m arksol demo --provider mock

# 跑完整对照实验并生成报告
python -m arksol eval --provider mock --out results/
```

想接真实模型（默认支持任何 OpenAI 兼容端点，含火山方舟/豆包）：

```bash
# 只走环境变量，绝不写进代码
export ARK_API_KEY=...
export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export ARK_MODEL=doubao-seed-2-1-turbo-260628
python -m arksol eval --provider openai-compat --out results/ark/
```

## 项目结构

```
ark-solution-lab/
├─ arksol/            核心包（零第三方依赖）
│  ├─ agents/         四个角色：intake / evidence / writer / critic
│  ├─ pipeline/       三条对照流水线：single / single_guarded / multi
│  ├─ llm/            Provider 抽象：mock（可注入故障）/ openai-compat
│  ├─ tools/          检索、成本换算
│  └─ kb/             示例知识库（每条标来源 URL + 读取日期）
├─ eval/              测试用例与评分口径
├─ tests/             unittest（标准库，无需 pytest）
├─ results/           跑分产物（README 里的数字都在这里复现）
└─ docs/              架构、选型取舍、评测方法
```

## 文档

| 文档 | 看什么 |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 架构、以及**为什么这么拆**（带出处） |
| [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) | 逐条选型取舍：我否掉了什么、为什么 |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | 指标定义、口径、局限与打假条件 |
| [`docs/大白话讲解.md`](docs/大白话讲解.md) | 零术语版本（第 7 步交付） |

## 设计时参考的证据

- Anthropic · [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)：多智能体约用 15× token；只在可高度并行、信息超出单窗口的任务上有明确收益
- Cognition · [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents) / [一年后复盘](https://cognition.com/blog/multi-agents-working)：写入保持单线程；无结构 swarm 是干扰项
- MAST · [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)：150–200+ 条轨迹归纳 14 种失败模式 / 3 大类

## License

MIT
