"""
光谱残差分析模块
评估NNLS光谱解混的质量
"""
from .spectral_residuals import main as spectral_residuals_dual

__all__ = ['spectral_residuals_dual']
