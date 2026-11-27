"""fresnelmix 核心 API（初始未实现，仅标准库）。"""


class FresnelMix:
    def __init__(self, ior):
        """构造"""
        raise NotImplementedError

    def f0(self):
        """法向反射率"""
        raise NotImplementedError

    def reflectance(self, cos_theta):
        """反射率"""
        raise NotImplementedError

    def mix(self, diffuse, specular, cos_theta):
        """混合"""
        raise NotImplementedError
