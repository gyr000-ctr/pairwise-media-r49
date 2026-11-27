# lightgrid

## 题面（可直接粘贴）

光照网格扩散（4 邻接迭代到收敛）。约束：收敛解与解析拉普拉斯解对照 ≤1e-6；边界格固定不参与更新（逐格不变 ≤1e-12）；迭代次数上限可配且收敛判据为两次差 ≤tol；常量初值与边界一致时结果不变（≤1e-12）；256x256 不超过 700 ms；非法网格可区分报错。 工作目录为仓库内的 lightgrid/。 验收：cd lightgrid && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd lightgrid && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
