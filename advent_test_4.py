# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, lsq_linear
from scipy.linalg import lstsq


# ========================== 阿伦方差模型 ==========================
def allan_deviation(tau, params):
    """
    计算阿伦方差模型
    参数:
        tau: 采样时间
        params: 包含4个噪声系数的参数数组 [A_wp, A_wf, A_ff, A_rw]
    返回:
        阿伦方差值
    """
    A_wp, A_wf, A_ff, A_rw = params

    return np.sqrt(
        (A_wp ** 2 / np.power(tau, 2)) +  # 白相位噪声项
        (A_wf ** 2 / tau) +  # 白频率噪声项
        (A_ff ** 2) +  # 闪烁频率噪声项
        (A_rw ** 2 * tau)  # 随机游走噪声项
    )



# ========================== 非线性方程组定义 ==========================
def equations(params, tau_meas, sigma_meas):
    """
    定义优化问题的残差方程
    参数:
        params: 模型参数
        tau_meas: 测量的tau值
        sigma_meas: 测量的sigma值
    返回:
        相对残差
    """
    model = allan_deviation(tau_meas, params)
    return (model - sigma_meas) / sigma_meas


def fit_allan_deviation(tau_measured, sigma_measured):
    """
    拟合阿伦方差参数
    参数:
        tau_measured: 测量的tau值数组
        sigma_measured: 测量的sigma值数组
    返回:
        优化后的参数数组 [A_wp, A_wf, A_ff, A_rw]
    """
    # 基于数据特征估算初始参数值
    A_wp_guess = 1.6e-13 * 0.1  # 根据最小tau处的数据估算白相位噪声
    A_rw_guess = 2.6e-13 / np.sqrt(1000)  # 根据最大tau处的数据估算随机游走
    initial_guess = np.array([A_wp_guess, 7e-14, 7.1e-14, A_rw_guess])

    # 使用Levenberg-Marquardt算法进行最小二乘拟合
    result = least_squares(
        equations,
        initial_guess,
        args=(tau_measured, sigma_measured),
        method='lm',
        max_nfev=100
    )
    return result.x


# ========================== 实验数据 ==========================
# USO实验测量的tau和sigma数据点
tau_measured = np.array([0.1, 1, 10, 100, 1000])
sigma_measured = np.array([1.6E-13, 7E-14, 7.1E-14, 8E-14, 2.6E-13])

# 调用拟合函数
params_opt = fit_allan_deviation(tau_measured, sigma_measured)

