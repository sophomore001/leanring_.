
from lisainstrument import noises
import numpy as np

from testsInstrument.advent_test_4 import fit_allan_deviation


def generate_noise_sequence(tau_measured, sigma_measured, fs=0.25, size=10000):
    """
    根据测量数据生成噪声序列

    参数:
        tau_measured: 测量时间数组
        sigma_measured: 测量阿伦方差数组
        fs: 采样频率(Hz)，默认为1
        size: 采样点数，默认为40000

    返回:
        合成的总噪声序列
    """
    # 使用fit_allan_deviation函数预测噪声参数
    estimated_params = fit_allan_deviation(tau_measured, sigma_measured)

    # 生成不同类型的噪声序列
    white_phase_noise = noises.white(fs, size, estimated_params[0])
    white_frequency_noise = noises.white(fs, size, estimated_params[1])
    flicker_frequency_noise = noises.pink(fs, size, estimated_params[2], fmin=1E-5)
    random_walk_noise = noises.red(fs, size, estimated_params[3])

    # 合成总噪声
    return white_phase_noise + white_frequency_noise + flicker_frequency_noise + random_walk_noise

