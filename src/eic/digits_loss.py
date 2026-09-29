import warnings
import numpy as np
from typing import Literal
from ..nd2py import nd2py as nd

warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")

__all__ = ['digits_loss', 'max_digits_loss']


class NoisyCalc(nd.NumpyCalc):
    def __init__(self, noise_level=1e-3, random_state=None, noisy_number=False, noisy_variable=False):
        super().__init__()
        self.noise_level = noise_level
        self.noisy_number = noisy_number
        self.noisy_variable = noisy_variable
        self.rng = np.random.default_rng(random_state)

        self.names_to_cache = [
            'visit_Add', 'visit_Sub', 'visit_Mul', 'visit_Div', 'visit_Pow', 
            'visit_Max', 'visit_Min', 'visit_Sin', 'visit_Cos', 'visit_Tan', 
            'visit_Sec', 'visit_Csc', 'visit_Cot', 'visit_Log', 'visit_LogAbs', 
            'visit_Exp', 'visit_Abs', 'visit_Neg', 'visit_Inv', 'visit_Sqrt', 
            'visit_SqrtAbs', 'visit_Pow2', 'visit_Pow3', 'visit_Arcsin', 'visit_Arccos', 
            'visit_Arctan', 'visit_Sinh', 'visit_Cosh', 'visit_Tanh', 'visit_Sech', 
            'visit_Csch', 'visit_Coth', 'visit_Sigmoid', 'visit_Regular', 'visit_Sour', 
            'visit_Targ', 'visit_Aggr', 'visit_Rgga', 'visit_Readout',
        ]
        # 之前实验发现，如果往 Number 和 Variable 中也加入噪声，会导致这个指标区分 PySR 和真实物理公式的能力变差，原因尚不明确，但先不加了
        if self.noisy_number:
            self.names_to_cache.append('visit_Number')
        if self.noisy_variable:
            self.names_to_cache.append('visit_Variable')

    def __getattribute__(self, name):
        visit_name = super().__getattribute__(name)
        if not name.startswith('visit_'):
            return visit_name

        if name not in self.names_to_cache:
            return visit_name

        def noisy_wrapper(*args, **kwargs):
            y = yield from visit_name(*args, **kwargs)
            n = self.rng.normal(0, 1, size=np.shape(y))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                noise = self.noise_level * y * n
                return y + noise

        return noisy_wrapper


# 这个算法在计算 f 的时候向每个中间节点添加指定信噪比的噪声，以模拟计算过程中保留有限位有效数字的影响
# 并通过信噪比估计最终的 f 的有效数字位数，输出相比于计算过程中所保留的有效数字位数的损失
def digits_loss(
    f: nd.Symbol,
    X: dict,
    noise_level=1e-6,
    random_state=0,
    eps=1e-6,
    return_type:Literal['dB', 'digits', 'raw']='digits',
    noisy_number=False,
    noisy_variable=False,
) -> float:
    """
    - f: nd2py function
    - z: a subformula of f to add noise
    - X: dict of variable arrays
    - noise_level: relative standard deviation of the noise
    - random_state: random seed
    - noisy_number: whether to add noise to constant numbers
    - noisy_variable: whether to add noise to input variables

    考虑数据 x 与噪声 e，x 的有效数字位数近似为 -log10(std(e/x)) = 0.05 * SNR (dB)
    其中 SNR (dB) = -10log(var(e/x))

    在计算过程中添加 e = noise_level * x * N(0,1) 的噪声，相当于保留了 -log10(noise_level) 位有效数字
    在有噪声和无噪声两种情况下计算 f，通过 SNR = -10log(var((f_noisy - f_true) / f_true)) 估计 f 的有效数字位数
    并返回 Delta SNR = SNR_f - SNR_x，表示 f 相比于 x 损失了多少有效数字位数
    """
    noisy_calc = NoisyCalc(
        noise_level=noise_level, 
        random_state=random_state, 
        noisy_number=noisy_number, 
        noisy_variable=noisy_variable
    )
    F_noisy = noisy_calc(f, X, use_eps=eps)
    F_clear = f.eval(X, use_eps=eps)
    nsr_x = noisy_calc.noise_level ** 2
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        nsr_f = np.nanvar((F_noisy - F_clear) / (F_clear + (F_clear == 0) * eps))
    delta = (nsr_f / nsr_x).clip(1e-30, 1e30)
    if return_type == 'digits':
        return 0.5 * np.log10(delta).item()
    elif return_type == 'dB':
        return -10 * np.log10(delta).item()
    else:
        return delta.item()


def max_digits_loss(f: nd.Symbol, X: dict):
    digits_loss_list = []
    for f_sub in f.iter_preorder():
        digits = digits_loss(f_sub, X, noise_level=1e-6, random_state=0, return_type='digits', eps=1e-6, noisy_number=False, noisy_variable=False)
        digits_loss_list.append(digits)
    # 有时因为一些意外原因，会出现 nsr_f / nsr_x = 0 的情况，
    # 此时 digits = 0.5 * np.log10(1e-30) = -15，
    # 显然不合理，故向上裁剪到 0
    return max(digits_loss_list + [0])  