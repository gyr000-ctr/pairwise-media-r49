"""lightgrid 核心 API（初始未实现，仅标准库）。"""


class LightGrid:
    def __init__(self, iters, tol):
        """构造"""
        raise NotImplementedError

    def solve(self, grid, fixed):
        """求解"""
        raise NotImplementedError

    def iterations(self):
        """实际迭代数"""
        raise NotImplementedError
