"""sssprofile 核心 API（初始未实现，仅标准库）。"""


class SssProfile:
    def __init__(self, sigma_a, sigma_s):
        """构造"""
        raise NotImplementedError

    def profile(self, r):
        """径向 profile"""
        raise NotImplementedError

    def normalize(self, samples):
        """归一化校验"""
        raise NotImplementedError
