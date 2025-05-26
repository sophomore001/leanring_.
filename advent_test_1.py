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
    # 使用绝对残差
    # return model - sigma_meas
    return (model - sigma_meas) / sigma_meas  # (这里两个残差各有优劣)


# 使用线性最小二乘法预测数据
def estimate_noise_parameters_old(taus, adev_measured):
    """
    基于Allan方差测量值的噪声参数估计
    :param taus: 积分时间数组
    :param adev_measured: 测量的Allan标准差数组
    :return: 估计的噪声参数字典（包含A_wp, A_wf, A_ff, A_rw）
    """
    fh = 1  # 高频截止频率

    # 构建设计矩阵 (根据Allan方差理论模型)
    # H 5x4
    H = np.vstack([
        np.sqrt(fh / np.array(taus) ** 2),  # 白相位噪声项 (1/τ²)
        np.sqrt(1 / np.array(taus)),  # 白频率噪声项 (1/τ)
        np.ones(len(taus)),  # 闪烁频率噪声项 (常数)
        np.sqrt(np.array(taus))  # 随机游走噪声项 (τ)
    ]).T
    print(f"设计矩阵: {H}", H.shape)
    # 计算矩阵条件数
    cond = np.linalg.cond(H)
    print(f"设计矩阵条件数: {cond:.2e}")
    # 构建目标向量 (平方后的测量值)
    # y 6x1
    y = adev_measured

    print(f"目标向量: {y}", y.shape)
    # 添加非负约束的最小二乘求解
    theta, _, _, _ = lstsq(H, y)
    print(f"最小二乘求解参数: {theta}", theta.shape)

    # 确保所有参数非负
    theta = np.maximum(theta, 0)
    print(f"最小二乘求解参数: {theta}", theta.shape)
    estimated_params = {
        'A_wp': theta[0],  # 白相位噪声幅度
        'A_wf': theta[1],  # 白频率噪声幅度
        'A_ff': theta[2],  # 闪烁频率噪声幅度
        'A_rw': theta[3]  # 随机游走噪声幅度
    }

    return estimated_params


# ========================== 实验数据 ==========================
# USO实验测量的tau和sigma数据点
# tau_measured = np.array([0.1, 1, 10, 100, 1000])
# sigma_measured = np.array([1.6E-13, 7E-14, 7.1E-14, 8E-14, 2.6E-13])
# 氢钟实验测量的tau和sigma数据点
tau_measured = np.array([1, 10, 100, 1000, 10000, 86400])
sigma_measured = np.array([2e-13, 3e-14, 6e-15, 3e-15, 2.5e-15, 2e-15])

# ========================== 调整初始猜测(氢钟) ==========================
# 基于氢钟数据特征估算初始参数值
A_wp_guess = 2e-13 * 1  # 根据最小tau=1s处的数据估算白相位噪声（最小τ=1s由wp主导）
A_rw_guess = 2e-15 / np.sqrt(86400)  # 根据最大tau=86400s处的数据估算随机游走（最大τ=86400s由rw主导）
# 假设τ=10s主导A_wf，τ=100s主导A_ff
initial_guess = np.array([A_wp_guess, 3e-14, 6e-15, A_rw_guess])


# ========================== 调整初始猜测 ==========================
# 基于数据特征估算初始参数值
# A_wp_guess = 1.6e-13 * 0.1  # 根据最小tau处的数据估算白相位噪声（最小τ=0.1由wp主导）
# A_rw_guess = 2.6e-13 / np.sqrt(1000)  # 根据最大tau处的数据估算随机游走（最大τ=1000由rw主导）
# 假设τ=1s主导A_wf，τ=10s主导A_ff
# initial_guess = np.array([A_wp_guess, 7e-14, 7.1e-14, A_rw_guess])

# ========================== 使用SciPy进行优化 ==========================
# 使用Levenberg-Marquardt算法进行最小二乘拟合
result = least_squares(
    equations,
    initial_guess,
    args=(tau_measured, sigma_measured),
    method='lm',  # Levenberg-Marquardt算法
    max_nfev=1000  # 最大迭代次数
)
# 提取优化后的参数
params_opt = result.x
# ============================使用最小二乘法估计参数并预测============================
estimated_params = estimate_noise_parameters_old(tau_measured, sigma_measured)

# ========================== 生成对比图 ==========================
# # 计算预测的Allan偏差
# sigma_sim = allan_deviation(tau_measured, params_opt)
# # 计算预测的Allan偏差（使用最小二乘法）
# sigma_sim_old = allan_deviation(tau_measured, estimated_params.values())
# 生成更密集的tau点以获得平滑的拟合曲线
tau_sim = np.logspace(-1, 4, 100)
sigma_sim = allan_deviation(tau_sim, params_opt)
sigma_sim_old = allan_deviation(tau_sim, estimated_params.values())

# 绘制测量数据和拟合模型的对比图
plt.figure(figsize=(10, 6))
# plt.loglog(tau_measured, sigma_measured, 'bo-', label='Measured (USO)')
plt.loglog(tau_measured, sigma_measured, 'bo-', label='Measured (hydrogen atomic)')
plt.loglog(tau_sim, sigma_sim, 'r--', label='Predicted (L-M)')
# plt.loglog(tau_measured, sigma_sim, 'r--', label='Predicted (L-M)')
plt.loglog(tau_sim, sigma_sim_old, 'g--', label='Predicted (Least Squares)')
# plt.loglog(tau_measured, sigma_sim_old, 'g--', label='Predicted (Least Squares)')
plt.xlabel('Averaging Time $\\tau$ (s)', fontsize=12)
plt.ylabel('Allan Deviation', fontsize=12)
# plt.title('Comparison of Allan Deviation: Measured (USO) vs Predicted (Model)', fontsize=14)
plt.title('Comparison of Allan Deviation: Measured (hydrogen atomic) vs Predicted (Model)', fontsize=14)
plt.grid(True, which='both', linestyle='--')
plt.legend()
plt.tight_layout()
plt.xlim(0.1, 1000)
plt.show()
# plt.savefig('uso_comparison.png', dpi=300)

# # Generate Allan deviation plots for four different noise components
# plt.figure(figsize=(12, 8))
#
# # Full model with all noise components
# sigma_full = allan_deviation(tau_sim, params_opt)
# plt.loglog(tau_sim, sigma_full, 'k-', label='Full model', linewidth=2)
#
# # White phase noise only
# params_wp = np.array([params_opt[0], 0, 0, 0])
# sigma_wp = allan_deviation(tau_sim, params_wp)
# plt.loglog(tau_sim, sigma_wp, 'r--', label='White phase noise only')
#
# # White frequency noise only
# params_wf = np.array([0, params_opt[1], 0, 0])
# sigma_wf = allan_deviation(tau_sim, params_wf)
# plt.loglog(tau_sim, sigma_wf, 'g--', label='White frequency noise only')
#
# # Flicker frequency noise only
# params_ff = np.array([0, 0, params_opt[2], 0])
# sigma_ff = allan_deviation(tau_sim, params_ff)
# plt.loglog(tau_sim, sigma_ff, 'b--', label='Flicker frequency noise only')
#
# # Random walk noise only
# params_rw = np.array([0, 0, 0, params_opt[3]])
# sigma_rw = allan_deviation(tau_sim, params_rw)
# plt.loglog(tau_sim, sigma_rw, 'm--', label='Random walk noise only')
#
# plt.xlabel('Averaging Time $\\tau$ (s)', fontsize=12)
# plt.ylabel('Allan Deviation', fontsize=12)
# plt.title('Contribution of Different Noise Components to Allan Deviation', fontsize=14)
# plt.grid(True, which='both', linestyle='--')
# plt.legend()
# plt.tight_layout()
# plt.xlim(0.1, 1000)
# plt.show()

1
