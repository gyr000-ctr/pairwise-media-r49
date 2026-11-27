# sssprofile

## 题面（可直接粘贴）

次表面散射偶极近似 profile。约束：profile 随距离单调递减；径向积分归一为 1（≤1e-9）；与解析偶极式逐点一致（≤1e-9）；平均自由程参数线性缩放 profile（≤1e-9）；10 万次求值不超过 200 ms；非法散射系数可区分报错。 工作目录为仓库内的 sssprofile/。 验收：cd sssprofile && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd sssprofile && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
