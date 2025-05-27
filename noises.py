#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Noises module.

Implements basic random noise generators, and use them to implement instrumental noises.

Authors:
    Jean-Baptiste Bayle <j2b.bayle@gmail.com>
"""

import logging
import numpy as np

from numpy import pi, sqrt
from lisaconstants import c

from .pyplnoise import pyplnoise

from clock_predict import *

logger = logging.getLogger(__name__)


def white(fs, size, asd):
    """Generate a white noise.

    Args:
        fs: sampling frequency [Hz]
        size: number of samples [samples]
        asd: amplitude spectral density [/sqrt(Hz)]
    """
    logger.debug("Generating white noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0
    generator = pyplnoise.WhiteNoise(fs, asd**2 / 2)
    return generator.get_series(size)


def powerlaw(fs, size, asd, alpha):
    """Generate a f^(alpha) noise in amplitude, with alpha > -1.

    Pyplnoise natively accepts alpha values between -1 and 0 (in amplitude).

    We extend the domain of validity to positive alpha values by generating noise time series
    corresponding to the nth-order antiderivative of the desired noise (with exponent alpha + n
    valid for direct generation with pyplnoise), and then taking its nth-order numerical derivative.

    When alpha is -1 (resp. 0), we use internally call the optimized `red()` function (resp.
    the `white()` function).

    Args:
        fs: sampling frequency [Hz]
        size: number of samples [samples]
        asd: amplitude spectral density [/sqrt(Hz)]
        alpha: frequency exponent in amplitude [alpha > -1 and alpha != 0]
    """
    logger.debug("Generating power-law noise (fs=%s Hz, size=%s, asd=%s, alpha=%s)", fs, size, asd, alpha)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0

    if alpha < -1:
        raise ValueError(f"invalid value for alpha '{alpha}', must be > -1.")
    if alpha == -1:
        return red(fs, size, asd)
    if -1 < alpha < 0:
        generator = pyplnoise.AlphaNoise(fs, fs / size, fs / 2, -2 * alpha)
        return asd / sqrt(2) * generator.get_series(size)
    if alpha == 0:
        return white(fs, size, asd)

    # Else, generate antiderivative and take numerical derivative
    antiderivative = powerlaw(fs, size, asd / (2 * pi), alpha - 1)
    return np.gradient(antiderivative, 1 / fs)

def violet(fs, size, asd):
    """Generate a violet noise in f in amplitude.

    Args:
        fs: sampling frequency [Hz]
        size: number of samples [samples]
        asd: amplitude spectral density [/sqrt(Hz)]
    """
    logger.debug("Generating violet noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0
    white_noise = white(fs, size, asd)
    return np.gradient(white_noise, 1 / fs) / (2 * pi)

# def blue(fs, size, asd):
#     """Generate a blue noise in f^0.5 in amplitude.
#         一般在进行钟差数据仿真时可以忽略相位闪烁噪声
#     Args:
#         fs: sampling frequency [Hz]
#         size: number of samples [samples]
#         asd: amplitude spectral density [/sqrt(Hz)]
#     """
#     logger.debug("Generating blue noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
#     if not asd:
#         logger.debug("Vanishing power spectral density, bypassing noise generation")
#         return 0
#
#     white_noise = white(fs, size, asd)
#
#     # 频域操作实现sqrt(f)关系
#     fft_data = np.fft.fft(white_noise)
#     freqs = np.fft.fftfreq(size, 1 / fs)
#
#     # 构造幅度调制函数
#     scaling = np.sqrt(np.abs(freqs))  # 核心数学关系
#     scaling[0] = 0  # 消除DC分量
#     scaling[scaling == np.inf] = 0  # 处理零频率
#
#     # 应用调制并逆变换
#     blue_noise = np.fft.ifft(fft_data * scaling).real
#
#     # 能量补偿因子 (保持与violet相同的校准逻辑)
#     return blue_noise * np.sqrt(2 * np.pi)



def pink(fs, size, asd, fmin=None):
    """生成粉红噪声,幅度谱密度正比于 f^(-1/2)

    粉红噪声的功率谱密度在频域上呈 1/f 关系,是一种常见的噪声类型。
    在时钟系统中,粉红噪声通常用来描述闪烁频率噪声。

    参数:
        fs: 采样频率 [Hz]
        size: 采样点数 [samples]
        asd: 幅度谱密度 [/sqrt(Hz)]
        fmin: 低频截止频率,默认为 fs/size [Hz]
            - 低于此频率的噪声将饱和
            - 用于避免低频发散
            - 通常取为频谱分辨率 fs/size

    返回:
        粉红噪声时间序列 [无量纲]
        - 序列长度为 size
        - 采样间隔为 1/fs
        - 幅度由 asd 缩放
    """
    logger.debug("Generating pink noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0

    """
    实际上 pink() 函数返回的也是无量纲的序列。解释如下:

    1. generator.get_series() 返回无量纲的随机序列
    2. asd 的单位是 [1/sqrt(Hz)]
    3. 但是 asd 在这里只是用来调节噪声强度的系数
    4. asd/sqrt(2) * get_series() 的结果仍然是无量纲的

    这样就能保持与 clock() 函数的一致性:
    - clock() 返回 ffd(分数频率偏差),是无量纲的
    - pink() 返回的序列也是无量纲的
    - 只是这个序列在频域上具有 f^(-1/2) 的特征
    - asd 参数用于控制噪声的整体强度

    所以之前注释中关于返回值单位的说明是错误的,
    pink() 函数返回的确实是无量纲的序列。
    """
    generator = pyplnoise.PinkNoise(fs, fmin or fs / size, fs / 2)
    return asd / sqrt(2) * generator.get_series(size)


def red(fs, size, asd):
    """Generate a red noise (also Brownian or random walk) in f^(-1) in amplitude.

    Args:
        fs: sampling frequency [Hz]
        size: number of samples [samples]
        asd: amplitude spectral density [/sqrt(Hz)]
    """
    logger.debug("Generating red noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0
    generator = pyplnoise.RedNoise(fs, fs / size)
    return asd / sqrt(2) * generator.get_series(size)


def infrared(fs, size, asd):
    """Generate an infrared noise in f^(-2) in amplitude.

    Args:
        fs: sampling frequency [Hz]
        size: number of samples [samples]
        asd: amplitude spectral density [/sqrt(Hz)]
    """
    logger.debug("Generating infrared noise (fs=%s Hz, size=%s, asd=%s)", fs, size, asd)
    if not asd:
        logger.debug("Vanishing power spectral density, bypassing noise generation")
        return 0
    red_noise = red(fs, size, asd)
    return np.cumsum(red_noise) * (2 * pi / fs)


def laser(fs, size, asd, shape):
    """Generate laser noise [Hz].

    This is a white noise with an infrared relaxation towards low frequencies,
    following the usual noise shape function,

        S_p(f) = asd^2 [ 1 + (fknee / f)^4 ]
               = asd^2 + asd^2 fknee^4 / f^4.

    The low-frequency part (infrared relaxation) can be disabled, in which
    case the noise shape becomes

        S_p(f) = asd^2.

    Args:
        asd: amplitude spectral density [Hz/sqrt(Hz)]
        fknee: cutoff frequency [Hz]
        shape: spectral shape, either 'white' or 'white+infrared'
    """
    fknee = 2E-3
    logger.debug(
        "Generating laser noise (fs=%s Hz, size=%s, asd=%s "
        "Hz/sqrt(Hz), fknee=%s Hz, shape=%s)",
        fs, size, asd, fknee, shape)

    if shape == 'white':
        return white(fs, size, asd)
    if shape == 'white+infrared':
        return white(fs, size, asd) + infrared(fs, size, asd * fknee**2)
    raise ValueError(f"invalid laser noise spectral shape '{shape}'")


def clock(fs, size, asd):
    """Generate clock noise fluctuations [ffd].
    生成时钟噪声波动 [ffd]

    The power spectral density in fractional frequency deviations is a pink noise,
    分数频率偏差的功率谱密度是粉红噪声,

    Clock noise saturates below 1E-5 Hz, as the low-frequency part is modeled by
    deterministing clock drifts.
    时钟噪声在1E-5 Hz以下饱和,因为低频部分由确定性时钟漂移建模。
    """
    """
        S_q(f) [ffd] = (asd)^2 f^(-1)
         Sq[ffd] 的单位分析:

    1. asd 的单位是 [1/sqrt(Hz)]
    2. f 的单位是 [Hz] 
    3. 代入公式 S_q(f) = (asd)^2 * f^(-1):
       - (asd)^2 的单位是 [1/Hz]
       - f^(-1) 的单位是 [1/Hz]
       - 所以 S_q(f) 的单位是 [1/Hz]
    
    这里需要区分:
    1. S_q(f) 是功率谱密度(PSD),单位是 [1/Hz]
    2. ffd 本身是分数频率偏差,是无量纲的
    3. asd 是幅度谱密度(ASD),单位是 [1/sqrt(Hz)]
    
    所以:
    - pink() 函数返回的 ffd 序列是无量纲的
    - 但描述这个序列的谱密度 S_q(f) 是有单位的
    - asd 作为输入参数用于调节噪声强度
    """
    """
    参数:
        fs: 采样频率 [Hz]
        size: 采样点数 [samples]
        asd: 幅度谱密度 [/sqrt(Hz)]

    返回:
        pink函数返回的是分数频率偏差(fractional frequency deviation, ffd)
        - 无量纲,表示频率偏差相对于载波频率的比值
        - 通常用于表征时钟的频率稳定性
        - 典型值在 10^-15 到 10^-12 量级
    """
    logger.debug("Generating clock noise fluctuations (fs=%s Hz, size=%s, asd=%s /sqrt(Hz))", fs, size, asd)
    return pink(fs, size, asd, fmin=1E-5)

# def new_clock(fs, size, asd):
#     """生成时钟噪声波动 [ffd].
#
#     包含五种噪声:
#     - 白相位噪声 (WP)
#     - 闪烁相位噪声 (FP)
#     - 白频率噪声 (WF)
#     - 闪烁频率噪声 (FF)
#     - 随机游走频率噪声 (RW)
#
#     Args:
#         fs: 采样频率 [Hz]
#         size: 采样点数 [samples]
#         asd: 幅度谱密度 [/sqrt(Hz)]
#     """
#     generator = ClockErrorGenerator(1/fs, f_h=fs/2)
#     noise_params = {
#         'A_wp': 1.5e-13,  # 白相位噪声系数
#         'A_fp': 5e-14,    # 闪烁相位噪声系数
#         'A_wf': 4e-14,    # 白频率噪声系数
#         'A_ff': 2e-15,    # 闪烁频率噪声系数
#         'A_rw': 3e-15/np.sqrt(86400)  # 随机游走频率噪声系数
#     }
#     y_total, _, _ = generator.generate_clock_errors(noise_params, size)
#     return y_total * asd * 1e-3  # 调整量级到与原函数一致

def modulation(fs, size, asd):
    """Generate modulation noise [ffd].

    The power spectral density as fractional frequency deviations reads

        S_M(f) [ffd] = (asd)^2 f^(2/3).

    It must be multiplied by the modulation frequency.

    Args:
        asd: amplitude spectral density [/sqrt(Hz)]
    """
    logger.debug("Generating modulation noise (fs=%s Hz, size=%s, asd=%s /sqrt(Hz))", fs, size, asd)
    return powerlaw(fs, size, asd, 1/3)


def backlink(fs, size, asd, fknee):
    """Generate backlink noise as fractional frequency deviation [ffd].

    The power spectral density in displacement is given by

        S_bl(f) [m] = asd^2 [ 1 + (fknee / f)^4 ].

    Multiplying by (2π f / c)^2 to express it as fractional frequency deviations,

        S_bl(f) [ffd] = (2π asd / c)^2 [ f^2 + (fknee^4 / f^2) ]
                      = (2π asd / c)^2 f^2 + (2π asd fknee^2 / c)^2 f^(-2)

    Because this is a optical pathlength noise expressed as fractional frequency deviation, it should
    be multiplied by the beam frequency to obtain the beam frequency fluctuations.

    Args:
        asd: amplitude spectral density [m/sqrt(Hz)]
        fknee: cutoff frequency [Hz]
    """
    logger.debug("Generating modulation noise (fs=%s Hz, size=%s, asd=%s m/sqrt(Hz), fknee=%s Hz)",
        fs, size, asd, fknee)
    return violet(fs, size, 2 * pi * asd / c) \
        + red(fs, size, 2 * pi * asd * fknee**2 / c)


def ranging(fs, size, asd):
    """Generate stochastic ranging noise [s].

    This is a white noise as a timing jitter,

        S_R(f) [s] = asd.

    Args:
        asd: amplitude spectral density [s/sqrt(Hz)]
    """
    logger.debug("Generating ranging noise (fs=%s Hz, size=%s, asd=%s s/sqrt(Hz))", fs, size, asd)
    return white(fs, size, asd)


def testmass(fs, size, asd, fknee, fbreak, frelax, shape):
    """Generate test-mass acceleration noise [m/s].

    Expressed in acceleration, the noise power spectrum reads

        S_delta(f) [ms^(-2)] =
            (asd)^2 [ 1 + (fknee / f)^2 ] [ 1 + (f / fbreak)^4)].

    Multiplying by 1 / (2π f)^2 yields the noise as a velocity,

        S_delta(f) [m/s] = (asd / 2π)^2 [ f^(-2) + (fknee^2 / f^4)
                           + f^2 / fbreak^4 + fknee^2 / fbreak^4 ]
                         = (asd fknee / 2π)^2 f^(-4)
                           + (asd / 2π)^2 f^(-2)
                           + (asd fknee / (2π fbreak^2))^2
                           + (asd / (2π fbreak^2)^2 f^2,

    which corresponds to the incoherent sum of an infrared, a red, a white,
    and a violet noise.

    A relaxation for more pessimistic models extending below the official LISA
    band of 1E-4 Hz can be added using the 'lowfreq-relax' shape, in which case
    the noise in acceleration picks up an additional f^(-4) term,

        S_delta(f) [ms^(-2)] = ... [ 1 + (frelax / f)^4 ].

    In velocity, this corresponds to additional terms,

        S_delta(f) [m/s] = ... [ 1 + (frelax / f)^4 ]
                         = ... + (asd fknee frelax^2 / 2π)^2 f^(-8)
                           + (asd frelax^2 / 2π)^2 f^(-6)
                           + (asd fknee frelax^2 / (2π fbreak^2))^2 f^(-4)
                           + (asd frelax^2 / (2π fbreak^2)^2 f^(-2).

    Args:
        asd: amplitude spectral density [ms^(-2)/sqrt(Hz)]
        fknee: low-frequency cutoff frequency [Hz]
        fbreak: high-frequency break frequency [Hz]
        frelax: low-frequency relaxation frequency [Hz]
        shape: spectral shape, either 'original' or 'lowfreq-relax'
    """
    logger.debug(
        "Generating test-mass noise (fs=%s Hz, size=%s, "
        "asd=%s ms^(-2)/sqrt(Hz), fknee=%s Hz, fbreak=%s Hz, "
        "frelax=%s Hz, shape=%s)",
        fs, size, asd, fknee, fbreak, frelax, shape
    )

    if shape == 'original':
        return (
            infrared(fs, size, asd * fknee / (2 * pi))
            + red(fs, size, asd / (2 * pi))
            + white(fs, size, asd * fknee / (2 * pi * fbreak**2))
            + violet(fs, size, asd / (2 * pi * fbreak**2))
        )
    if shape == 'lowfreq-relax':
        # We need to integrate infrared noises to get f^(-6) and f^(-8) noises
        # Start with f^(-4) noises
        relaxation1 = infrared(fs, size, asd * frelax**2 / (2 * pi))
        relaxation2 = infrared(fs, size, asd * fknee * frelax**2 / (2 * pi))
        # Integrate once for f^(-6)
        relaxation1 = np.cumsum(relaxation1) * (2 * pi / fs)
        relaxation2 = np.cumsum(relaxation2) * (2 * pi / fs)
        # Integrate twice for f^(-8)
        relaxation2 = np.cumsum(relaxation2) * (2 * pi / fs)
        # Add the other components to the original noise
        infrared_asd = asd * fknee * np.sqrt(1 + (frelax / fbreak)**4) / (2 * pi)
        red_asd = asd * np.sqrt(1 + (frelax / fbreak)**4) / (2 * pi)
        return (
            relaxation2 # f^(-8)
            + relaxation1 # f^(-6)
            + infrared(fs, size, infrared_asd)
            + red(fs, size, red_asd)
            + white(fs, size, asd * fknee / (2 * pi * fbreak**2))
            + violet(fs, size, asd / (2 * pi * fbreak**2))
        )
    raise ValueError(f"invalid test-mass noise spectral shape '{shape}'")

def oms(fs, size, asd, fknee):
    """Generate optical metrology system (OMS) noise allocation [ffd].

    The power spectral density in displacement is given by

        S_oms(f) [m] = asd^2 [ 1 + (fknee / f)^4 ].

    Multiplying by (2π f / c)^2 to express it as fractional frequency deviations,

        S_oms(f) [ffd] = (2π asd / c)^2 [ f^2 + (fknee^4 / f^2) ]
                       = (2π asd / c)^2 f^2 + (2π asd fknee^2 / c)^2 f^(-2).

    Note that the level of this noise depends on the interferometer and the type of beatnote.

    Warning: this corresponds to the overall allocation for the OMS noise from the Performance
    Model. It is a collection of different noises, some of which are duplicates of standalone
    noises we already implement in the simulation (e.g., backlink noise).

    Args:
        asd: amplitude spectral density [m/sqrt(Hz)]
        fknee: cutoff frequency [Hz]
    """
    logger.debug("Generating OMS noise (fs=%s Hz, size=%s, asd=%s m/sqrt(Hz), fknee=%s Hz)",
        fs, size, asd, fknee)
    return violet(fs, size, 2 * pi * asd / c) \
        + red(fs, size, 2 * pi * asd * fknee**2 / c)

def jitter(fs, size, asd, fknee):
    """Generate jitter for one angular degree of freedom.

    The power spectral density in angle is given by

        S_jitter(f) [rad] = asd^2 [ 1 + (fknee / f)^4 ],

    which is converted to angular velocity by mutliplying by (2π f)^2,

        S_jitter(f) [rad/s] = (2π asd)^2 [ f^2 + (fknee^4 / f^2) ]
                            = (2π asd)^2 f^2 + (2π asd fknee^2)^2 f^(-2).

    Args:
        asd: amplitude spectral density [rad/sqrt(Hz)]
        fknee: cutoff frequency [Hz]
    """
    logger.debug("Generating jitter (fs=%s Hz, size=%s, asd=%s rad/sqrt(Hz), fknee=%s Hz)",
        fs, size, asd, fknee)
    return violet(fs, size, 2 * pi * asd) \
        + red(fs, size, 2 * pi * asd * fknee**2)

def dws(fs, size, asd):
    """Generate DWS measurement noise.

    The power spectral density in angle is given by

        S_dws(f) [rad] = asd^2,

    which is converted to angular velocity by mutliplying by (2π f)^2,

        S_dws(f) [rad/s] = (2π asd)^2 f^2.

    Args:
        asd: amplitude spectral density [rad/sqrt(Hz)]
    """
    logger.debug("Generating DWS measurement (fs=%s Hz, size=%s, asd=%s rad/sqrt(Hz))", fs, size, asd)
    return violet(fs, size, 2 * pi * asd)

def moc_time_correlation(fs, size, asd):
    """MOC time correlation noise.

    High-level noise model for the uncertainty we have in computing the MOC
    time correlation (or time couples), i.e., the equivalent TCB times for the
    equally-sampled TPS timestamps.

    Assumed to be a white noise in timing,

        S_moc(f) [s] = asd^2.

    Args:
        asd: amplitude spectral density [s/sqrt(Hz)]
    """
    logger.debug("Generating MOC time correlation noise (fs=%s Hz, size=%s, "
                 "asd=%s s/sqrt(Hz))", fs, size, asd)
    return white(fs, size, asd)
