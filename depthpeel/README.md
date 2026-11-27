# depthpeel

## 题面（可直接粘贴）

深度剥离与层面积统计。约束：每层取当前最近且严格递增（≤1e-12）；层数按面积阈值自适应终止；必须输出每层覆盖面积，且各层面积之和等于首层面积（≤1e-9）；剥离结果与「排序后按阈值切分」的参考实现逐层一致（≤1e-12）；空输入返回空；1 万三角形不超过 400 ms；非法层数与负深度可区分报错。 工作目录为仓库内的 depthpeel/。 验收：cd depthpeel && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd depthpeel && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
