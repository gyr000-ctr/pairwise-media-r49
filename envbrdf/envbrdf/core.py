"""envbrdf 核心 API（初始未实现，仅标准库）。"""


class EnvBrdf:
    def __init__(self, table):
        """构造"""
        raise NotImplementedError

    def lookup(self, n_dot_v, roughness):
        """查表"""
        raise NotImplementedError

    def fwhm(self, roughness):
        """高光半高宽"""
        raise NotImplementedError
