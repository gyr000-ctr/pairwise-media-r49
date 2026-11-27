# fresnelmix

## 题面（可直接粘贴）

菲涅尔反射（复数折射率与偏振）。约束：必须分别给出 s/p 偏振分量的反射率，且与精确菲涅尔公式差 ≤1e-9；在全内反射临界角之外两者都返回 1（≤1e-12）；Schlick 近似只允许作为加速分支并与精确式差 ≤1e-3；法向反射率满足 F0=((n-1)/(n+1))^2（≤1e-12）；混合权重和为 1（≤1e-12）；10 万次求值不超过 150 ms；非法折射率与越界余弦可区分报错。 工作目录为仓库内的 fresnelmix/。 验收：cd fresnelmix && python -m pytest tests -q。 实现只允许使用 Python 标准库。

## 运行方式

```bash
cd fresnelmix && python -m pytest tests -q
```

初始 15 failed（实现前）。仅使用 Python 标准库。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
