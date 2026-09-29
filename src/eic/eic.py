import functools
import signal
import numpy as np
from  ..nd2py import nd2py as nd
from ..nd2py.nd2py.core.base_visitor import yield_nothing


# Decorator to unpack operands for operations
# This allows us to handle operations with multiple operands in a clean way
# We can also use this decorator to suppress numpy errors
def unpack_operands():
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, node, *args, **kwargs):
            # Calculate the values of the operands
            yield from yield_nothing()
            X = []
            for op in node.operands:
                clear_x, noisy_x = yield (op, args, kwargs)
                X.append((clear_x, noisy_x))
            clear_X, noisy_X = zip(*X)
            # Use the defined 'visit_<Operation>' as 'func' to process the operands
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return func(self, node, *clear_X, *noisy_X, *args, **kwargs)

        return wrapper

    return decorator


class EICCalc(nd.NumpyCalc):
    def __init__(
        self, 
        noise_level=1e-3,
        random_state=None, 
        noisy_number=False, 
        noisy_variable=False,
        use_eps: float = 0.0,
        calc_EIC_for_all_subtree=False,
    ):
        super().__init__()
        self.noise_level = noise_level
        self.noisy_number = noisy_number
        self.noisy_variable = noisy_variable
        self.use_eps = use_eps
        self.calc_EIC_for_all_subtree = calc_EIC_for_all_subtree
        self.rng = np.random.default_rng(random_state)
        self.eic = -np.inf
    
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

    def __call__( self, node: nd.Symbol, vars: dict = {} ):
        clear_y, noisy_y = super().__call__( node, vars=vars )
        self.update_eic(clear_y, noisy_y)
        return max(self.eic, 0)

    def add_noise(self, y):
        n = self.rng.normal(0, 1, size=np.shape(y))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            noise = self.noise_level * y * n
            return y + noise
    
    def update_eic(self, clear_y, noisy_y):
        eps = self.use_eps
        nsr_x = self.noise_level ** 2
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            nsr_f = np.nanvar((clear_y - noisy_y) / (clear_y + eps * (clear_y == 0)))
        delta = (nsr_f / nsr_x).clip(1e-30, 1e30)
        eic = 0.5 * np.log10(delta).item()
        if eic > self.eic:
            self.eic = eic
    
    def generic_visit(self, node: nd.Symbol, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__}.visit_{type(node).__name__} not implemented"
        )

    def visit_Number(self, node: nd.Number, *args, **kwargs):
        yield from yield_nothing()
        clear_y = np.asarray(node.value)
        if self.noisy_number:
            noisy_y = self.add_noise(clear_y)
        else:
            noisy_y = clear_y
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    def visit_Variable(self, node: nd.Variable, *args, **kwargs):
        yield from yield_nothing()
        clear_y = np.asarray(kwargs["vars"][node.name])
        if self.noisy_variable:
            noisy_y = self.add_noise(clear_y)
        else:
            noisy_y = clear_y
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Add(self, node: nd.Add, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = clear_x1 + clear_x2
        noisy_y = self.add_noise(noisy_x1 + noisy_x2)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sub(self, node: nd.Sub, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = clear_x1 - clear_x2
        noisy_y = self.add_noise(noisy_x1 - noisy_x2)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Mul(self, node: nd.Mul, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = clear_x1 * clear_x2
        noisy_y = self.add_noise(noisy_x1 * noisy_x2)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Div(self, node: nd.Div, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        eps = self.use_eps
        clear_y = clear_x1 / (clear_x2 + eps * (clear_x2 == 0))
        noisy_y = self.add_noise(noisy_x1 / (noisy_x2 + eps * (noisy_x2 == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Pow(self, node: nd.Pow, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = clear_x1 ** (clear_x2)
        noisy_y = self.add_noise(noisy_x1 ** noisy_x2)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Max(self, node: nd.Max, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = np.maximum(clear_x1, clear_x2)
        noisy_y = self.add_noise(np.maximum(noisy_x1, noisy_x2))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Min(self, node: nd.Min, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        clear_y = np.minimum(clear_x1, clear_x2)
        noisy_y = self.add_noise(np.minimum(noisy_x1, noisy_x2))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sin(self, node: nd.Sin, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.sin(clear_x)
        noisy_y = self.add_noise(np.sin(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Cos(self, node: nd.Cos, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.cos(clear_x)
        noisy_y = self.add_noise(np.cos(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Tan(self, node: nd.Tan, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.tan(clear_x)
        noisy_y = self.add_noise(np.tan(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sec(self, node: nd.Sec, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.cos(clear_x) + eps * (np.cos(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.cos(noisy_x) + eps * (np.cos(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Csc(self, node: nd.Csc, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.sin(clear_x) + eps * (np.sin(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.sin(noisy_x) + eps * (np.sin(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Cot(self, node: nd.Cot, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.tan(clear_x) + eps * (np.tan(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.tan(noisy_x) + eps * (np.tan(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Log(self, node: nd.Log, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = np.log(clear_x + eps * (clear_x == 0))
        noisy_y = self.add_noise(np.log(noisy_x + eps * (noisy_x == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_LogAbs(self, node: nd.LogAbs, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = np.log(np.abs(clear_x) + eps * (clear_x == 0))
        noisy_y = self.add_noise(np.log(np.abs(noisy_x) + eps * (noisy_x == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Exp(self, node: nd.Exp, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.exp(clear_x)
        noisy_y = self.add_noise(np.exp(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Abs(self, node: nd.Abs, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.abs(clear_x)
        noisy_y = self.add_noise(np.abs(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Neg(self, node: nd.Neg, clear_x, noisy_x, *args, **kwargs):
        clear_y = -clear_x
        noisy_y = self.add_noise(-noisy_x)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Inv(self, node: nd.Inv, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (clear_x + eps * (clear_x == 0))
        noisy_y = self.add_noise(1 / (noisy_x + eps * (noisy_x == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sqrt(self, node: nd.Sqrt, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.sqrt(clear_x)
        noisy_y = self.add_noise(np.sqrt(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_SqrtAbs(self, node: nd.SqrtAbs, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.sqrt(np.abs(clear_x))
        noisy_y = self.add_noise(np.sqrt(np.abs(noisy_x)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Pow2(self, node: nd.Pow2, clear_x, noisy_x, *args, **kwargs):
        clear_y = clear_x ** 2
        noisy_y = self.add_noise(noisy_x ** 2)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Pow3(self, node: nd.Pow3, clear_x, noisy_x, *args, **kwargs):
        clear_y = clear_x ** 3
        noisy_y = self.add_noise(noisy_x ** 3)
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Arcsin(self, node: nd.Arcsin, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.arcsin(clear_x)
        noisy_y = self.add_noise(np.arcsin(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Arccos(self, node: nd.Arccos, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.arccos(clear_x)
        noisy_y = self.add_noise(np.arccos(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Arctan(self, node: nd.Arctan, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.arctan(clear_x)
        noisy_y = self.add_noise(np.arctan(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sinh(self, node: nd.Sinh, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.sinh(clear_x)
        noisy_y = self.add_noise(np.sinh(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Cosh(self, node: nd.Cosh, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.cosh(clear_x)
        noisy_y = self.add_noise(np.cosh(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Tanh(self, node: nd.Tanh, clear_x, noisy_x, *args, **kwargs):
        clear_y = np.tanh(clear_x)
        noisy_y = self.add_noise(np.tanh(noisy_x))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sech(self, node: nd.Sech, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.cosh(clear_x) + eps * (np.cosh(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.cosh(noisy_x) + eps * (np.cosh(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Csch(self, node: nd.Csch, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.sinh(clear_x) + eps * (np.sinh(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.sinh(noisy_x) + eps * (np.sinh(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Coth(self, node: nd.Coth, clear_x, noisy_x, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (np.tanh(clear_x) + eps * (np.tanh(clear_x) == 0))
        noisy_y = self.add_noise(1 / (np.tanh(noisy_x) + eps * (np.tanh(noisy_x) == 0)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Sigmoid(self, node: nd.Sigmoid, clear_x, noisy_x, *args, **kwargs):
        clear_y = 1 / (1 + np.exp(-clear_x))
        noisy_y = self.add_noise(1 / (1 + np.exp(-noisy_x)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

    @unpack_operands()
    def visit_Regular(self, node: nd.Regular, clear_x1, clear_x2, noisy_x1, noisy_x2, *args, **kwargs):
        eps = self.use_eps
        clear_y = 1 / (1 + (np.abs(clear_x1) + eps * (clear_x1 == 0)) ** (-clear_x2))
        noisy_y = self.add_noise(1 / (1 + (np.abs(noisy_x1) + eps * (noisy_x1 == 0)) ** (-noisy_x2)))
        if self.calc_EIC_for_all_subtree:
            self.update_eic(clear_y, noisy_y)
        return clear_y, noisy_y

# class TimeoutException(Exception): 
#     pass

# def handler(signum, frame): 
#     raise TimeoutException("Timeout")

# signal.signal(signal.SIGALRM, handler)

def get_eic(
        f: nd.Symbol,
        X: dict, 
        timeout=None, 
        noise_level=1e-6,
        random_state=None,
        noisy_number=False,
        noisy_variable=False,
        use_eps: float = 1e-6,
        calc_EIC_for_all_subtree=True,
    ) -> float:
    """
    计算公式 f 在数据 X 上的 EIC 指标
    - f: nd2py function
    - X: dict of variable arrays
    - kwargs: arguments for EICCalc

    返回 f 的 EIC 指标，表示 f 相比于输入变量 X 损失了多少有效数字位数
    """
    calc = EICCalc(
        noise_level=noise_level, 
        random_state=random_state, 
        noisy_number=noisy_number, 
        noisy_variable=noisy_variable,
        use_eps=use_eps,
        calc_EIC_for_all_subtree=calc_EIC_for_all_subtree,
    )

    return calc(f, X)
    # try:
    #     if timeout is not None:
    #         signal.alarm(timeout)
    #     return calc(f, X)
    # except TimeoutException as e:
    #     print("Timed out!")
    #     return calc.eic if np.isfinite(calc.eic) else np.nan
