# envbrdf

## 题面（可直接粘贴）

环境 BRDF 查表与解析对照。约束：表内双线性插值与解析 GGX 近似式对照 ≤1e-9；边界 clamp 不越界；积分能量 ≤1（单次散射能量守恒）；粗糙度单调时高光变宽（半高宽单调不减）；10 万次查表不超过 200 ms；非法表与越界粗糙度可区分报错。 工作目录为仓库内的 envbrdf/。 验收：cd envbrdf && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd envbrdf && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
