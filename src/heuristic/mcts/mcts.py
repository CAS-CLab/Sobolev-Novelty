import json
import time
import random
import logging
import hashlib
import os
from collections import OrderedDict
import sklearn
import traceback
import numpy as np
import sympy as sp
import pandas as pd
from dataclasses import replace
from pathlib import Path
from tqdm import tqdm
from typing import List, Generator, Tuple, Dict
from .utils import preprocess, sample_Xy
from .sobolev_guidance import sobolev_penalty, stable_state_id
from ...eic.eic import get_eic
from ...nd2py import nd2py as nd
from ...nd2py.nd2py.utils import seed_all, Timer, NamedTimer, R2_score, RMSE_score
from ...sobolev import (
    PruningConfig,
    RefitGeometryHint,
    RefitResult,
    ShortlistConfig,
    SobolevConfig,
    SobolevEvaluator,
    TermEvaluationCache,
    TermSpec,
    prune_and_refit,
    rank_by_base,
    rank_by_structural,
)
from ...sobolev.decomposition import (
    decompose_expand_mul,
    parse_expression,
    to_project_expression_string,
)
from ...sobolev.signature import evaluate_sympy, select_geometry_indices

def simplify(eq:nd.Symbol):
    try:
        expr = sp.parse_expr(eq.to_str())
        expr = sp.simplify(expr)
        return nd.parse_expr(str(expr))
    except:
        return eq.copy()


class Node:
    def __init__(self, eqtrees:List[nd.Symbol]):
        # Formula part
        self.eqtrees = eqtrees
        self.phi = None
        self.complexity = None
        self.reward = None
        self.base_reward = None
        self.final_reward = None
        self.r2 = None
        self.eic = None
        self.fit_type = None
        self.sobolev_penalty = 0.0
        self.sobolev_success = None
        self.sobolev_failure_type = None
        self.sobolev_failure_message = None
        self.sobolev_search_result = None
        self.sobolev_full_result = None
        self.fitted_expression_before_pruning = None
        self.pruned_expression = None
        self.pruning_result = None
        self.pruning_trace_id = None
        self.base_reward_before_pruning = None
        self.coefficient_fitting_time = 0.0
        self.eic_time = 0.0
        self.candidate_runtime = 0.0
        self.sobolev_parent_geometry_key = None
        self.sobolev_parent_candidate = None
        self.candidate_id = None
        self.sobolev_evaluated = False
        self.selection_stage = None
        self.base_rank = None
        self.structural_rank = None
        self.shortlist_status = None
        self._elite_train_idx = None
        self._elite_eval_idx = None
        self._elite_preparsed_hint = None
        self._elite_additive_coefficients = None

        # MC Tree part
        self.parent = None
        self.children = []
        self.N = 0
        self.Q = 0
    
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return '[' + ', '.join(str(eq) for eq in self.eqtrees) + ']' + f' (N={self.N}, Q={self.Q/(self.N+1e-6):.2f})'

    def UCT(self, c) -> float:
        if self.parent is None: return float('inf')
        return self.Q/(self.N+1e-6) + c * np.sqrt(np.log(self.parent.N) / (self.N+1e-6))

    def to_route(self, N=5, c=1.41) -> str:
        """
        Root
        ├ Node1
        ┆ ├ self
        ┆ └ Node1-2
        └ Node2
        """
        rev_route = [self]
        tmp = self
        while tmp.parent:
            rev_route.append(tmp.parent)
            tmp = tmp.parent
        items = []
        for node in rev_route:
            if node.parent:
                siblings = node.parent.children
                UCT = {x: x.UCT(c) for x in siblings}
                siblings = sorted(siblings, key=UCT.get, reverse=True)
                siblings = siblings[:N]
            else:
                siblings = [node]
                UCT = {node: 0.0}
            new_items = [f'{node} (UCT={UCT[node]:.2f})' for node in siblings]
            self_idx = siblings.index(node)
            for idx, item in enumerate(items): 
                items[idx] = ('├ ' if idx < len(items)-1 else '└ ') + item.replace('\n', '\n' + ('┆ ' if idx < len(items)-1 else '  '))
            new_items[self_idx] = '\033[31m' + new_items[self_idx] + '\033[0m' + ('\n' if items else '') + '\n'.join(items)
            items = new_items
        assert len(items) == 1
        return items[0]

    def copy(self) -> 'Node':
        copy = Node([eqtree.copy() for eqtree in self.eqtrees])
        copy.phi = self.phi
        copy.complexity = self.complexity
        copy.reward = self.reward
        copy.base_reward = self.base_reward
        copy.final_reward = self.final_reward
        copy.r2 = self.r2
        copy.eic = self.eic
        copy.fit_type = self.fit_type
        copy.sobolev_penalty = self.sobolev_penalty
        copy.sobolev_success = self.sobolev_success
        copy.sobolev_failure_type = self.sobolev_failure_type
        copy.sobolev_failure_message = self.sobolev_failure_message
        copy.sobolev_search_result = self.sobolev_search_result
        copy.sobolev_full_result = self.sobolev_full_result
        copy.fitted_expression_before_pruning = self.fitted_expression_before_pruning
        copy.pruned_expression = self.pruned_expression
        copy.pruning_result = self.pruning_result
        copy.pruning_trace_id = self.pruning_trace_id
        copy.base_reward_before_pruning = self.base_reward_before_pruning
        copy.coefficient_fitting_time = self.coefficient_fitting_time
        copy.eic_time = self.eic_time
        copy.candidate_runtime = self.candidate_runtime
        copy.sobolev_parent_geometry_key = self.sobolev_parent_geometry_key
        copy.sobolev_parent_candidate = self.sobolev_parent_candidate
        copy.candidate_id = self.candidate_id
        copy.sobolev_evaluated = self.sobolev_evaluated
        copy.selection_stage = self.selection_stage
        copy.base_rank = self.base_rank
        copy.structural_rank = self.structural_rank
        copy.shortlist_status = self.shortlist_status
        copy._elite_train_idx = self._elite_train_idx
        copy._elite_eval_idx = self._elite_eval_idx
        copy._elite_preparsed_hint = self._elite_preparsed_hint
        copy._elite_additive_coefficients = self._elite_additive_coefficients
        return copy

    def adopt_evaluation(self, other: 'Node') -> None:
        """Adopt the winning fitted realization while preserving this state."""

        for name in (
            'r2', 'phi', 'complexity', 'eic', 'reward', 'base_reward',
            'final_reward', 'fit_type', 'sobolev_penalty', 'sobolev_success',
            'sobolev_failure_type', 'sobolev_failure_message',
            'sobolev_search_result', 'sobolev_full_result',
            'fitted_expression_before_pruning', 'pruned_expression',
            'pruning_result', 'pruning_trace_id', 'base_reward_before_pruning',
            'coefficient_fitting_time', 'eic_time', 'candidate_runtime',
            'candidate_id', 'sobolev_evaluated', 'selection_stage',
            'base_rank', 'structural_rank', 'shortlist_status',
            '_elite_train_idx', '_elite_eval_idx', '_elite_preparsed_hint',
            '_elite_additive_coefficients',
        ):
            setattr(self, name, getattr(other, name))
        

class MCTS(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Monte Carlo Tree Search from the sample side, only use f but not g
    """
    def __init__(self, 
        binary:List[str|nd.Symbol]=[nd.Add, nd.Sub, nd.Mul, nd.Div, nd.Max, nd.Min],
        unary:List[str|nd.Symbol]=[nd.Sqrt, nd.Log, nd.Abs, nd.Neg, nd.Inv, nd.Sin, nd.Cos, nd.Tan],
        leaf:List[float|nd.Number]=[nd.Number(1), nd.Number(0.5)],
        const_range=None,
        child_num=50,
        n_playout=100,
        d_playout=10,
        max_len=30,
        c=1.41,
        n_iter=100,
        time_limit=None,
        time_limit_min_iterations=None,
        time_floor_snapshot_seconds=None,
        sample_num=300,
        log_per_iter=float('inf'),
        log_per_sec=float('inf'),
        save_path=None,
        keep_vars=False,
        normalize_y=False,
        normalize_X=False,
        remove_abnormal=False,
        random_state=42,
        ratio=1.0,
        eta=0.999,
        xi=1.0,
        alpha=0.0,
        max_var=10,
        use_digits_loss=False,
        r2_influenced_alpha=False,
        structural_metric=None,
        sobolev_alpha=0.0,
        sobolev_tau=1/np.sqrt(10),
        sobolev_lambda_value=1.0,
        sobolev_lambda_gradient=1.0,
        geometry_sample_size=None,
        sobolev_cache=True,
        sobolev_fast_gram=True,
        sobolev_detailed_logging=False,
        sobolev_min_valid_samples=32,
        sobolev_gram_condition_threshold=1e6,
        sobolev_failure_policy='max_penalty',
        sobolev_dataset_identity='dataset',
        sobolev_candidate_log_path=None,
        sobolev_failure_log_path=None,
        sobolev_geometry_indices_path=None,
        sobolev_elite_log_path=None,
        sobolev_run_summary_path=None,
        sobolev_pruning=False,
        sobolev_max_prunes=0,
        sobolev_acceptance_tolerance=0.0,
        sobolev_pruning_trace_path=None,
        sobolev_evaluator_mode='full',
        sobolev_pruning_geometry_reuse=False,
        sobolev_selection_mode='all',
        shortlist_mode='hybrid',
        shortlist_size=64,
        shortlist_ratio=0.03,
        shortlist_min=32,
        shortlist_max=128,
        prune_elite_k=5,
        sobolev_reranking_trace_path=None,
        sobolev_iteration_log_path=None,
        eic_random_state=None,
        **kwargs):
        self.max_var = max_var

        self.eqtree = None
        self.binary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in binary]
        self.unary = [eval(x, globals(), nd.__dict__) if isinstance(x, str) else x for x in unary]
        self.leaf = [nd.Number(x) if isinstance(x, float) else x for x in leaf]
        self.variables = []

        self.const_range = const_range
        self.child_num = child_num
        self.n_playout = n_playout
        self.d_playout = d_playout
        self.max_len = max_len
        self.c = c
        self.n_iter = n_iter
        self.sample_num = sample_num

        self.log_per_iter = log_per_iter
        self.log_per_sec = log_per_sec
        self.records = []
        self.logger = logging.getLogger(__name__)
        self.step_timer = Timer()
        self.view_timer = Timer()
        self.named_timer = NamedTimer()
        self.save_path = save_path
        self.keep_vars = keep_vars
        self.normalize_y = normalize_y
        self.normalize_X = normalize_X
        self.remove_abnormal = remove_abnormal
        self.random_state = random_state
        self.ratio = ratio
        self.time_limit = time_limit
        if time_limit_min_iterations is not None:
            time_limit_min_iterations = int(time_limit_min_iterations)
            if time_limit_min_iterations < 1:
                raise ValueError('time_limit_min_iterations must be positive when supplied')
            if time_limit is None:
                raise ValueError('time_limit_min_iterations requires time_limit')
        if time_floor_snapshot_seconds is not None:
            time_floor_snapshot_seconds = float(time_floor_snapshot_seconds)
            if time_floor_snapshot_seconds <= 0:
                raise ValueError('time_floor_snapshot_seconds must be positive when supplied')
        self.time_limit_min_iterations = time_limit_min_iterations
        self.time_floor_snapshot_seconds = time_floor_snapshot_seconds
        self.time_floor_snapshot = None
        self.eta = eta
        self.xi = xi
        self.alpha = alpha
        self.use_digits_loss = use_digits_loss
        self.r2_influenced_alpha = r2_influenced_alpha
        if structural_metric is None:
            structural_metric = 'eic' if (alpha > 0.0 or xi < 1.0) else 'none'
        if structural_metric not in {'none', 'eic', 'sobolev', 'eic+sobolev'}:
            raise ValueError(f'Unknown structural_metric: {structural_metric}')
        if sobolev_alpha < 0:
            raise ValueError('sobolev_alpha must be non-negative')
        if sobolev_failure_policy not in {'max_penalty', 'invalid'}:
            raise ValueError(f'Unknown sobolev_failure_policy: {sobolev_failure_policy}')
        if sobolev_max_prunes < 0:
            raise ValueError('sobolev_max_prunes must be non-negative')
        if sobolev_acceptance_tolerance < 0:
            raise ValueError('sobolev_acceptance_tolerance must be non-negative')
        if sobolev_evaluator_mode not in {'full', 'incremental'}:
            raise ValueError(f'Unknown sobolev_evaluator_mode: {sobolev_evaluator_mode}')
        if sobolev_selection_mode not in {'all', 'elite'}:
            raise ValueError(f'Unknown sobolev_selection_mode: {sobolev_selection_mode}')
        self.structural_metric = structural_metric
        self.sobolev_alpha = sobolev_alpha
        self.sobolev_tau = sobolev_tau
        self.sobolev_lambda_value = sobolev_lambda_value
        self.sobolev_lambda_gradient = sobolev_lambda_gradient
        self.geometry_sample_size = geometry_sample_size
        self.sobolev_cache = sobolev_cache
        self.sobolev_fast_gram = sobolev_fast_gram
        self.sobolev_detailed_logging = sobolev_detailed_logging
        self.sobolev_min_valid_samples = sobolev_min_valid_samples
        self.sobolev_gram_condition_threshold = sobolev_gram_condition_threshold
        self.sobolev_failure_policy = sobolev_failure_policy
        self.sobolev_dataset_identity = sobolev_dataset_identity
        self.sobolev_candidate_log_path = sobolev_candidate_log_path
        self.sobolev_failure_log_path = sobolev_failure_log_path
        self.sobolev_geometry_indices_path = sobolev_geometry_indices_path
        self.sobolev_elite_log_path = sobolev_elite_log_path
        self.sobolev_run_summary_path = sobolev_run_summary_path
        self.sobolev_pruning = sobolev_pruning
        self.sobolev_max_prunes = sobolev_max_prunes
        self.sobolev_acceptance_tolerance = sobolev_acceptance_tolerance
        self.sobolev_pruning_trace_path = sobolev_pruning_trace_path
        self.sobolev_evaluator_mode = sobolev_evaluator_mode
        self.sobolev_pruning_geometry_reuse = sobolev_pruning_geometry_reuse
        self.sobolev_selection_mode = sobolev_selection_mode
        self.shortlist_config = ShortlistConfig(
            mode=shortlist_mode,
            size=shortlist_size,
            ratio=shortlist_ratio,
            minimum=shortlist_min,
            maximum=shortlist_max,
            prune_elite_k=prune_elite_k,
        )
        self.sobolev_reranking_trace_path = sobolev_reranking_trace_path
        self.sobolev_iteration_log_path = sobolev_iteration_log_path
        self.eic_random_state = eic_random_state
        self.sobolev_evaluator = None
        self.sobolev_geometry_indices = None
        self.candidate_counter = 0
        self.current_iter = 0
        self.sobolev_evaluation_count = 0
        self.sobolev_success_count = 0
        self.sobolev_failure_counts = {}
        self.sobolev_penalties = []
        self.sobolev_algorithm_counts = {}
        self.sobolev_total_evaluator_time = 0.0
        self.sobolev_geometry_cache_hit_count = 0
        self.sobolev_incremental_hit_count = 0
        self.sobolev_incremental_fallback_counts = {}
        self.sobolev_full_parse_count = 0
        self.sobolev_full_raw_geometry_count = 0
        self.sobolev_module_parse_hits = 0
        self.sobolev_module_parse_misses = 0
        self.sobolev_module_parse_cache = OrderedDict()
        self.sobolev_module_parse_cache_max_entries = 50_000
        self.sobolev_pool_candidate_count = 0
        self.sobolev_shortlist_evaluation_count = 0
        self.sobolev_base_rejected_count = 0
        self.sobolev_elite_prune_count = 0
        self.sobolev_elite_iteration_count = 0
        self.sobolev_iteration_traces = []
        self.base_fit_eic_total_seconds = 0.0
        self.pruning_candidates_attempted = 0
        self.pruning_candidates_changed = 0
        self.pruning_accepted_count = 0
        self.pruning_rejected_count = 0
        self.pruning_termination_counts = {}
        self.pipeline_failure_counts = {}
        self.unique_raw_expressions = set()
        self.unique_fitted_expressions = set()
        config_payload = json.dumps({
            'structural_metric': structural_metric,
            'sobolev_alpha': sobolev_alpha,
            'sobolev_tau': sobolev_tau,
            'sobolev_lambda_value': sobolev_lambda_value,
            'sobolev_lambda_gradient': sobolev_lambda_gradient,
            'geometry_sample_size': geometry_sample_size,
            'sobolev_cache': sobolev_cache,
            'sobolev_fast_gram': sobolev_fast_gram,
            'sobolev_min_valid_samples': sobolev_min_valid_samples,
            'sobolev_gram_condition_threshold': sobolev_gram_condition_threshold,
            'sobolev_failure_policy': sobolev_failure_policy,
            'sobolev_pruning': sobolev_pruning,
            'sobolev_max_prunes': sobolev_max_prunes,
            'sobolev_acceptance_tolerance': sobolev_acceptance_tolerance,
            'sobolev_evaluator_mode': sobolev_evaluator_mode,
            'sobolev_pruning_geometry_reuse': sobolev_pruning_geometry_reuse,
            'sobolev_selection_mode': sobolev_selection_mode,
            'shortlist_mode': shortlist_mode,
            'shortlist_size': shortlist_size,
            'shortlist_ratio': shortlist_ratio,
            'shortlist_min': shortlist_min,
            'shortlist_max': shortlist_max,
            'prune_elite_k': prune_elite_k,
            'random_state': random_state,
            'eic_random_state': eic_random_state,
            'time_limit_min_iterations': time_limit_min_iterations,
            'time_floor_snapshot_seconds': time_floor_snapshot_seconds,
        }, sort_keys=True)
        self.configuration_hash = hashlib.sha256(config_payload.encode()).hexdigest()[:16]

        if kwargs:
            self.logger.warning('Unknown args: %s', ', '.join(f'{k}={v}' for k,v in kwargs.items()))

    def __repr__(self):
        res = 'None' if self.eqtree is None else self.eqtree.to_str()
        return '{}({})'.format(self.__class__.__name__, res)

    def _time_limit_reached(self, iteration:int, elapsed:float) -> bool:
        """Return whether both the wall-time and optional iteration floor hold."""
        if self.time_limit is None or elapsed <= self.time_limit:
            return False
        return (
            self.time_limit_min_iterations is None
            or iteration >= self.time_limit_min_iterations
        )

    def fit(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray], y, n_iter=None, use_tqdm=False, 
            early_stop:callable=lambda r2, complexity, eq: r2 > 0.99999):
        """
        Args:
            X: (n_samples, n_dims)
            y: (n_samples,)
        """
        seed_all(self.random_state)
        n_iter = n_iter or self.n_iter

        # Preprocess
        X = preprocess(X)
        X, y = sample_Xy(X, y, self.sample_num)
        self._initialize_sobolev(X)
        self.variables = [nd.Variable(var) for var in X]
        self.keep_vars = len(X) if self.keep_vars and len(X) <= self.max_var / 2 else 0

        # Root Node
        variables = list(X.keys())
        if len(variables) > self.max_var: variables = variables[:self.max_var]
        self.MC_tree = Node([nd.Variable(var) for var in variables])
        self.pareto_front = []

        # Search
        stop = False
        self.best = None
        self.start_time = time.time()
        for iter in tqdm(range(1, n_iter+1), disable=not use_tqdm):
            self.current_iter = iter
            iteration_started = time.perf_counter()
            iteration_snapshot = {
                'candidate_variants': self.candidate_counter,
                'sobolev_evaluations': self.sobolev_evaluation_count,
                'full_raw_geometry': self.sobolev_full_raw_geometry_count,
                'geometry_cache_hits': self.sobolev_geometry_cache_hit_count,
                'incremental_hits': self.sobolev_incremental_hit_count,
                'shortlist_evaluations': self.sobolev_shortlist_evaluation_count,
                'pool_candidates': self.sobolev_pool_candidate_count,
                'pruning_attempted': self.pruning_candidates_attempted,
                'evaluator_seconds': self.sobolev_total_evaluator_time,
                'base_fit_eic_seconds': self.base_fit_eic_total_seconds,
            }
            record = {'iter': iter, 'time': time.time() - self.start_time}
            log = {'Iter': iter}

            leaf = self.select(self.MC_tree)
            expand = self.expand(leaf, X, y)
            reward, best_simulated = self.simulate(expand, X, y)
            self.backpropagate(expand, reward)
            
            self.step_timer.add(1)

            if self.best is None or best_simulated.reward > self.best.reward:
                self.best = best_simulated.copy()
                if not self._elite_is_enabled():
                    self.set_reward(self.best, X, y)
                self.eqtree = simplify(self.best.phi)
                record['complexity'] = self.best.complexity
                record['reward'] = self.best.reward
                record['r2'] = self.best.r2
                record['eqtree'] = str(self.best)
                stop = early_stop(self.best.r2, self.best.complexity, self.best.phi)

            record.update({
                'wall_time': time.time() - self.start_time,
                'iteration_seconds': time.perf_counter() - iteration_started,
                'candidate_variants_delta': (
                    self.candidate_counter - iteration_snapshot['candidate_variants']
                ),
                'candidate_variants_cumulative': self.candidate_counter,
                'sobolev_evaluations_delta': (
                    self.sobolev_evaluation_count
                    - iteration_snapshot['sobolev_evaluations']
                ),
                'sobolev_evaluations_cumulative': self.sobolev_evaluation_count,
                'full_raw_geometry_delta': (
                    self.sobolev_full_raw_geometry_count
                    - iteration_snapshot['full_raw_geometry']
                ),
                'geometry_cache_hits_delta': (
                    self.sobolev_geometry_cache_hit_count
                    - iteration_snapshot['geometry_cache_hits']
                ),
                'incremental_hits_delta': (
                    self.sobolev_incremental_hit_count
                    - iteration_snapshot['incremental_hits']
                ),
                'shortlist_evaluations_delta': (
                    self.sobolev_shortlist_evaluation_count
                    - iteration_snapshot['shortlist_evaluations']
                ),
                'pool_candidates_delta': (
                    self.sobolev_pool_candidate_count
                    - iteration_snapshot['pool_candidates']
                ),
                'pruning_attempted_delta': (
                    self.pruning_candidates_attempted
                    - iteration_snapshot['pruning_attempted']
                ),
                'sobolev_evaluator_seconds_delta': (
                    self.sobolev_total_evaluator_time
                    - iteration_snapshot['evaluator_seconds']
                ),
                'base_fit_eic_seconds_delta': (
                    self.base_fit_eic_total_seconds
                    - iteration_snapshot['base_fit_eic_seconds']
                ),
                'best_reward': self._finite_or_none(self.best.reward),
                'best_r2': self._finite_or_none(self.best.r2),
                'best_complexity': self._finite_or_none(self.best.complexity),
                'best_expression': str(self.best.phi),
                'best_min_novelty': (
                    self.best.sobolev_search_result.min_novelty
                    if self.best.sobolev_search_result is not None else None
                ),
                'best_mean_novelty': (
                    self.best.sobolev_search_result.mean_novelty
                    if self.best.sobolev_search_result is not None else None
                ),
            })
            # Lightweight resource telemetry for externally supervised runs.
            # These counters do not affect selection, fitting, or RNG state.
            term_cache = (
                self.sobolev_evaluator.cache
                if self.sobolev_evaluator is not None else None
            )
            geometry_cache = (
                self.sobolev_evaluator.geometry_cache
                if self.sobolev_evaluator is not None else None
            )
            record.update({
                'sobolev_cache_memory_bytes': (
                    term_cache.memory_bytes if term_cache is not None else 0
                ),
                'sobolev_geometry_cache_memory_bytes': (
                    geometry_cache.memory_bytes if geometry_cache is not None else 0
                ),
                'sobolev_geometry_cache_entries': (
                    geometry_cache.entry_count if geometry_cache is not None else 0
                ),
            })

            # Keep the last *fully completed* iteration at or before the
            # requested wall-clock checkpoint.  This is deliberately a floor
            # snapshot: an iteration that starts before but finishes after the
            # checkpoint is not attributed to the checkpoint budget.
            if (
                self.time_floor_snapshot_seconds is not None
                and record['wall_time'] <= self.time_floor_snapshot_seconds
            ):
                self.time_floor_snapshot = dict(record)

            if (not iter % self.log_per_iter) or (self.step_timer.time > self.log_per_sec) or (iter == n_iter) or (stop):
                record['speed'] = (str(self.step_timer), str(self.view_timer))
                # record['detailed_time'] = str(self.named_timer)
                
                log['Reward'] = f'{self.best.reward:.5f}'
                log['Complexity'] = self.best.complexity
                log['R2'] = f'{self.best.r2:.5f}'
                log['Best'] = str(self.best)
                log['Best equation'] = str(self.best.phi)
                log['Speed'] = f'{record['speed'][0]} ({record['speed'][1]})'
                # log['Time'] = record['detailed_time']
                log['Current'] = str(expand)
                self.logger.info(' | '.join(f'\033[4m{k}\033[0m: {v}' for k, v in log.items()))
                self.step_timer.clear()

            self.records.append(record)
            if self.save_path:
                with open(self.save_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

            if stop:
                self.logger.note(f'Early stopping at iter {iter} with R2 {self.best.r2} ({self.best.phi})')
                self._finalize_run(X)
                return 'early_stop'

            elapsed = time.time() - self.start_time
            if self._time_limit_reached(iter, elapsed):
                minimum_note = (
                    '' if self.time_limit_min_iterations is None
                    else f' and minimum iteration {self.time_limit_min_iterations}'
                )
                self.logger.note(
                    f'Time limit {self.time_limit}s{minimum_note} reached at iter '
                    f'{iter} with R2 {self.best.r2} ({self.best.phi})'
                )
                self._finalize_run(X)
                return 'time_limit'

        self._finalize_run(X)
        return 'iter_limit'
        # self.logger.info(expand.to_route(3, self.c))

    def predict(self, X:np.ndarray|pd.DataFrame|Dict[str,np.ndarray]) -> np.ndarray:
        """
        Args:
            X: (n_samples, n_dims)
        Returns:
            y: (n_samples,)
        """
        if self.eqtree is None: raise ValueError('Model not fitted yet')
        X = preprocess(X)
        pred = self.eqtree.eval(X)
        pred[~np.isfinite(pred)] = 0
        return pred

    def action(self, state:Node, action:Tuple[nd.Symbol,int]) -> Node:
        """
        用 action[0] 取代 state.eqtrees[action[1]]
        """
        parent_geometry_key = (
            state.sobolev_search_result.candidate_geometry_key
            if state.sobolev_search_result is not None
            else state.sobolev_parent_geometry_key
        )
        parent_candidate = state
        state = state.copy()
        state.sobolev_parent_geometry_key = parent_geometry_key
        state.sobolev_parent_candidate = (
            parent_candidate if self._elite_is_enabled() else None
        )
        eqtree, idx = action
        if idx == len(state.eqtrees): 
            state.eqtrees.append(eqtree)
        elif isinstance(eqtree, nd.Empty):
            state.eqtrees.pop(idx)
        else:
            state.eqtrees[idx] = eqtree
        return state

    def check_valid_action(self, state:Node, action:Tuple[nd.Symbol,int]) -> bool:
        eqtree, idx = action
        if idx > min(len(state.eqtrees) + 1, self.max_var): return False
        if idx == len(state.eqtrees) and isinstance(eqtree, nd.Empty): return False
        if len(state.eqtrees) == 1 and isinstance(eqtree, nd.Empty): return False
        if sum(len(eqtree) for i, eqtree in enumerate(state.eqtrees) if i != idx) + len(eqtree) > self.max_len: return False
        if idx < self.keep_vars: return False
        return True

    def iter_valid_action(self, state:Node, shuffle=False) -> Generator[Tuple[nd.Symbol,int],None,None]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]

        eqtree_loader = []
        for sym in self.binary:
            if sym in [nd.Add, nd.Mul, nd.Max, nd.Min]: # Abelian group
                for i in range(len(leafs)):
                    for j in range(i, len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
            elif sym in [nd.Sub, nd.Div]: # Non-abelian group
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        if i != j:
                            eqtree_loader.append(sym(leafs[i], leafs[j]))
            else:
                for i in range(len(leafs)):
                    for j in range(len(leafs)):
                        eqtree_loader.append(sym(leafs[i], leafs[j]))
        for sym in self.unary:
            for i in range(len(leafs)):
                eqtree_loader.append(sym(leafs[i]))
        for sym in self.variables:
            eqtree_loader.append(sym)
        eqtree_loader.append(nd.Empty())

        idx_loader = list(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))

        loader = [(eqtree, idx) for eqtree in eqtree_loader for idx in idx_loader]
        if shuffle:
            random.shuffle(loader)

        for eqtree, idx in loader:
            if self.check_valid_action(state, (eqtree, idx)):
                yield eqtree, idx
    
    def pick_valid_action(self, state:Node) -> Tuple[nd.Symbol,int]:
        leafs = [*state.eqtrees, *self.variables, *self.leaf]
        for _ in range(1000):
            op = random.choice(self.binary + self.unary + self.variables + [nd.Empty()])
            idx = random.choice(range(self.keep_vars, min(len(state.eqtrees) + 1, self.max_var)))
            if isinstance(op, type): op = op(*random.choices(leafs, k=op.n_operands))
            if self.check_valid_action(state, (op, idx)): break
        else:
            raise ValueError('Cannot find valid action')
        return op, idx

    def select(self, root:Node) -> Node:
        node = root
        while node.children:
            node = max(node.children, key=lambda x: x.Q/(x.N+1e-6) + self.c * np.sqrt(np.log(node.N+1) / (x.N+1e-6)))
            # node = max(node.children, key=lambda x: x.Q/(x.N+1e-6))
        return node

    def expand(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Node:
        for idx, action in enumerate(self.iter_valid_action(node, shuffle=True)):
            child = self.action(node, action)
            child.parent = node
            child.xchild = len(node.children)
            node.children.append(child)
            if self.child_num and idx + 1 >= self.child_num: break
        if not node.children: return node  # leaf node
        return random.choice(node.children)

    def simulate(self, node:Node, X:Dict[str,np.ndarray], y:np.ndarray) -> Tuple[float, Node]:
        if self._elite_is_enabled():
            return self._simulate_elite(node, X, y)
        self.set_reward(node, X, y)
        best = node
        for i in range(self.n_playout):
            state = node
            for j in range(self.d_playout):
                action = self.pick_valid_action(state)
                if action is None: break
                state = self.action(state, action)
                self.set_reward(state, X, y)
                if state.reward > best.reward: best = state
        return best.reward, best

    def _simulate_elite(
        self,
        node:Node,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
    ) -> Tuple[float, Node]:
        """Base-screen one natural rollout pool, then Sobolev-rerank its shortlist."""

        pool = []
        self.set_reward(node, X, y, defer_sobolev=True)
        pool.append(node)
        for _ in range(self.n_playout):
            state = node
            for _ in range(self.d_playout):
                action = self.pick_valid_action(state)
                if action is None:
                    break
                state = self.action(state, action)
                self.set_reward(state, X, y, defer_sobolev=True)
                pool.append(state)
        selected = self._select_elite_candidate(pool, X, y)
        return selected.reward, selected

    def backpropagate(self, node:Node, reward:float):
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent

    def get_reward(self, complexity:int, r2:float, eic:float) -> float:
        """
        reward = eta ** compmlexity * xi ** eic / (2 - r2) - alpha * eic
        """
        reward = 1 / (2 - r2)
        if self.eta < 1.0:
            reward *= self.eta ** complexity
        if self.structural_metric in {'eic', 'eic+sobolev'}:
            if self.xi < 1.0:
                reward *= self.xi ** eic
            if self.alpha > 0.0:
                if self.r2_influenced_alpha:
                    reward -= self.alpha * (1 - r2) * eic
                else:
                    reward -= self.alpha * eic
        return reward

    def set_reward(
        self,
        node:Node,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
        defer_sobolev:bool = False,
    ) -> float:
        self.view_timer.add(1)
        candidate_start = time.perf_counter()

        if self.ratio < 1.0:
            train_idx = np.random.rand(y.shape[0]) < self.ratio
            eval_idx = ~train_idx
        else:
            train_idx = np.ones_like(y).astype(bool)
            eval_idx = train_idx

        # Calculate Z
        Z = np.zeros((y.shape[0], 1+len(node.eqtrees)))
        Z[:, 0] = 1.0
        for idx, eqtree in enumerate(node.eqtrees, 1):
            try:
                Z[:, idx] = eqtree.eval(X)
            except:
                Z[:, idx] = np.nan
        Z[~np.isfinite(Z)] = 0

        # linear model as phi
        try:
            fit_start = time.perf_counter()
            assert np.isfinite(Z).all()
            A, _, _, _ = np.linalg.lstsq(Z[train_idx, :], y[train_idx], rcond=None)
            A = np.round(A, 6)
            node.r2 = R2_score(y[eval_idx], Z[eval_idx, :] @ A)
            node.phi = nd.Number(A[0]) if A[0] != 0 else None
            for a, op in zip(A[1:], node.eqtrees):
                if a == 0: pass
                elif a == 1: 
                    if node.phi is None: node.phi = op
                    else: node.phi += op
                elif a == -1:
                    if node.phi is None: node.phi = -op
                    else: node.phi -= op
                else: 
                    if node.phi is None: node.phi = nd.Number(a) * op
                    else: node.phi += nd.Number(a) * op
            if node.phi is None: node.phi = nd.Number(0.0)
            node.complexity = len(node.phi)
            coefficient_fitting_time = time.perf_counter() - fit_start
            eic_start = time.perf_counter()
            node.eic = get_eic(node.phi, X, random_state=self.eic_random_state)
            eic_time = time.perf_counter() - eic_start
            node.reward = self.get_reward(node.complexity, node.r2, node.eic)
            node.fit_type = 'additive'
            self._finalize_candidate(
                node, X, y, train_idx, eval_idx,
                candidate_start, coefficient_fitting_time, eic_time,
                preparsed_hint=(
                    None
                    if defer_sobolev
                    else self._prepare_additive_hint(A, node.eqtrees, tuple(X.keys()))
                ),
                defer_sobolev=defer_sobolev,
                deferred_additive_coefficients=A if defer_sobolev else None,
            )
        except Exception as e:
            # self.logger.warning(traceback.format_exc())
            node.r2 = -np.inf
            node.complexity = np.inf
            node.reward = 0.0
            self._write_pipeline_failure(node, 'additive_candidate_evaluation', e)


        # prod model as phi: y = phi(f1, f2, ...) = a0 * |f1|^a1 * |f2|^a2 * ...
        try:
            node2 = node.copy()
            product_start = time.perf_counter()
            Z_ = np.log(np.abs(Z).clip(1e-10, None))
            Z_[:, 0] = 1.0
            y_ = np.log(np.abs(y).clip(1e-10, None))
            assert np.isfinite(Z_).all() and np.isfinite(y_).all()
            A, _, _, _ = np.linalg.lstsq(Z_[train_idx, :], y_[train_idx], rcond=None)
            A[0] = np.exp(A[0])
            A = np.round(A, 6)
            prod = 1
            for z, a in zip(Z[:, 1:].T, A[1:]):
                prod *= np.abs(z) ** a
            A[0] *= np.sign(np.median(y[train_idx] / (A[0] * prod[train_idx]).clip(1e-6)))
            node2.r2 = R2_score(y[eval_idx], A[0] * prod[eval_idx])
            node2.phi = nd.Number(A[0]) if A[0] != 1 else None
            for idx, (a, op) in enumerate(zip(A[1:], node.eqtrees), 1):
                if (Z[idx]<0).any(): op = nd.Abs(op)
                if a == 0: pass
                elif a == 1: 
                    if node2.phi is None: node2.phi = op
                    else: node2.phi *= op
                elif a == -1: 
                    if node2.phi is None: node2.phi = nd.Inv(op)
                    else: node2.phi /= op
                elif a == 2:
                    if node2.phi is None: node2.phi = nd.Pow2(op)
                    else: node2.phi *= nd.Pow2(op)
                elif a == -2:
                    if node2.phi is None: node2.phi = nd.Inv(nd.Pow2(op))
                    else: node2.phi /= nd.Pow2(op)
                elif a == 3:
                    if node2.phi is None: node2.phi = nd.Pow3(op)
                    else: node2.phi *= nd.Pow3(op)
                elif a == -3:
                    if node2.phi is None: node2.phi = nd.Inv(nd.Pow3(op))
                    else: node2.phi /= nd.Pow3(op)
                elif a == 0.5:
                    if node2.phi is None: node2.phi = nd.Sqrt(op)
                    else: node2.phi *= nd.Sqrt(op)
                elif a == -0.5:
                    if node2.phi is None: node2.phi = nd.Inv(nd.Sqrt(op))
                    else: node2.phi /= nd.Sqrt(op)
                elif a > 0:
                    if node2.phi is None: node2.phi = op ** nd.Number(a)
                    else: node2.phi *= op ** nd.Number(a)
                elif a < 0:
                    if node2.phi is None: node2.phi = nd.Inv(op ** nd.Number(-a))
                    else: node2.phi /= op ** nd.Number(-a)
                else: raise ValueError(f'Unknown a: {a}')
            if node2.phi is None: node2.phi = nd.Number(1.0)
            node2.complexity = len(node2.phi)
            product_fitting_time = time.perf_counter() - product_start
            product_eic_start = time.perf_counter()
            node2.eic = get_eic(node2.phi, X, random_state=self.eic_random_state)
            product_eic_time = time.perf_counter() - product_eic_start
            node2.reward = self.get_reward(node2.complexity, node2.r2, node2.eic)
            node2.fit_type = 'product'
            self._finalize_candidate(
                node2, X, y, train_idx, eval_idx,
                candidate_start, product_fitting_time, product_eic_time,
                defer_sobolev=defer_sobolev,
            )
            if node2.reward > node.reward:
                node.adopt_evaluation(node2)
        except Exception as e:
            self._write_pipeline_failure(node, 'product_candidate_evaluation', e)
            # self.logger.warning(traceback.format_exc())
            # logger.warning(str(e))
            # node.r2 = -np.inf
            # node.complexity = np.inf
            # node.reward = 0.0

        # update pareto front
        dominated = []
        for i in self.pareto_front:
            if (i.r2 >= node.r2 and i.complexity <= node.complexity):
                break
            if (i.r2 <= node.r2 and i.complexity >= node.complexity):
                dominated.append(i)
        else:
            self.pareto_front = [i for i in self.pareto_front if i not in dominated]
            self.pareto_front.append(node)
            self.pareto_front = sorted(self.pareto_front, key=lambda x: (x.r2, -x.complexity))

        if not np.isfinite(node.reward): node.reward = 0.0

    def _initialize_sobolev(self, X:Dict[str,np.ndarray]) -> None:
        if not self._sobolev_is_required():
            return
        config = SobolevConfig(
            lambda_value=self.sobolev_lambda_value,
            lambda_gradient=self.sobolev_lambda_gradient,
            threshold=self.sobolev_tau,
            min_valid_samples=self.sobolev_min_valid_samples,
            fast_gram=self.sobolev_fast_gram,
            gram_condition_threshold=self.sobolev_gram_condition_threshold,
            cache_enabled=self.sobolev_cache,
            geometry_sample_size=self.geometry_sample_size,
            geometry_seed=self.random_state,
            candidate_geometry_cache=self.sobolev_evaluator_mode == 'incremental',
            output_scale_free_internal=self.sobolev_evaluator_mode == 'incremental',
            parent_child_incremental=self.sobolev_evaluator_mode == 'incremental',
        )
        self.sobolev_evaluator = SobolevEvaluator(
            config,
            TermEvaluationCache(enabled=self.sobolev_cache),
        )
        n_samples = len(next(iter(X.values())))
        self.sobolev_geometry_indices = select_geometry_indices(
            n_samples,
            self.sobolev_dataset_identity,
            self.random_state,
            self.geometry_sample_size,
        )
        if self.sobolev_geometry_indices_path:
            path = Path(self.sobolev_geometry_indices_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.sobolev_geometry_indices)

    def _finalize_candidate(
        self,
        node:Node,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
        train_idx:np.ndarray,
        eval_idx:np.ndarray,
        candidate_start:float,
        coefficient_fitting_time:float,
        eic_time:float,
        preparsed_hint:RefitGeometryHint|None = None,
        defer_sobolev:bool = False,
        deferred_additive_coefficients:np.ndarray|None = None,
    ) -> None:
        raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
        node.fitted_expression_before_pruning = str(node.phi)
        node.pruned_expression = None
        node.pruning_result = None
        node.pruning_trace_id = None
        node.base_reward = node.reward
        node.base_reward_before_pruning = node.base_reward
        node.coefficient_fitting_time = coefficient_fitting_time
        node.eic_time = eic_time
        self.base_fit_eic_total_seconds += coefficient_fitting_time + eic_time
        node._elite_train_idx = np.array(train_idx, dtype=bool, copy=True)
        node._elite_eval_idx = np.array(eval_idx, dtype=bool, copy=True)
        node._elite_preparsed_hint = preparsed_hint
        node._elite_additive_coefficients = (
            np.array(deferred_additive_coefficients, dtype=float, copy=True)
            if deferred_additive_coefficients is not None else None
        )
        node.sobolev_evaluated = False
        node.selection_stage = 'base_pool' if defer_sobolev else 'all_candidates'
        node.shortlist_status = None
        result = None
        if self._sobolev_is_required() and not defer_sobolev:
            result = self._evaluate_sobolev_expression(
                node, raw_expression, node.phi, X,
                coefficient_fitting_time, eic_time,
                preparsed_hint=preparsed_hint,
            )
            node.sobolev_search_result = result
            if self.sobolev_pruning and self.sobolev_max_prunes > 0 and result.success:
                result = self._prune_candidate(
                    node, raw_expression, result, X, y, train_idx, eval_idx,
                )
            node.sobolev_search_result = result
            node.sobolev_success = result.success
            node.sobolev_failure_type = result.failure_type.value
            node.sobolev_failure_message = result.failure_message
            node.sobolev_evaluated = True
        if defer_sobolev:
            node.sobolev_penalty = None
            node.reward = node.base_reward
            node.sobolev_success = None
        elif self.structural_metric in {'sobolev', 'eic+sobolev'}:
            node.sobolev_penalty = (
                sobolev_penalty(result.term_novelties, self.sobolev_tau)
                if result is not None and result.success
                else 1.0
            )
            if self.sobolev_alpha == 0.0:
                node.reward = node.base_reward
            elif (result is None or not result.success) and self.sobolev_failure_policy == 'invalid':
                node.reward = 0.0
            else:
                node.reward = node.base_reward - self.sobolev_alpha * node.sobolev_penalty
        else:
            node.sobolev_penalty = 0.0
            node.reward = node.base_reward
            if not self._sobolev_is_required():
                node.sobolev_success = None
        node.final_reward = node.reward
        node.candidate_runtime = time.perf_counter() - candidate_start
        if node.sobolev_penalty is not None:
            self.sobolev_penalties.append(node.sobolev_penalty)
        raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
        self.unique_raw_expressions.add(raw_expression)
        self.unique_fitted_expressions.add(str(node.phi))
        self.candidate_counter += 1
        node.candidate_id = self.candidate_counter
        self._write_candidate(node)

    def _elite_is_enabled(self) -> bool:
        return self.sobolev_selection_mode == 'elite' and self._sobolev_is_required()

    def _apply_deferred_sobolev(
        self,
        node:Node,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
        *,
        allow_pruning:bool,
    ) -> None:
        """Evaluate one shortlisted base candidate without changing pool semantics."""

        raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
        parent = node.sobolev_parent_candidate
        if parent is not None and parent.sobolev_search_result is not None:
            node.sobolev_parent_geometry_key = (
                parent.sobolev_search_result.candidate_geometry_key
            )
        preparsed_hint = node._elite_preparsed_hint
        if (
            preparsed_hint is None
            and node.fit_type == 'additive'
            and node._elite_additive_coefficients is not None
        ):
            preparsed_hint = self._prepare_additive_hint(
                node._elite_additive_coefficients,
                node.eqtrees,
                tuple(X.keys()),
            )
            node._elite_preparsed_hint = preparsed_hint
        result = self._evaluate_sobolev_expression(
            node,
            raw_expression,
            node.phi,
            X,
            node.coefficient_fitting_time,
            node.eic_time,
            preparsed_hint=preparsed_hint,
        )
        node.sobolev_search_result = result
        node.sobolev_success = result.success
        node.sobolev_failure_type = result.failure_type.value
        node.sobolev_failure_message = result.failure_message
        node.sobolev_evaluated = True
        if (
            allow_pruning
            and self.sobolev_pruning
            and self.sobolev_max_prunes > 0
            and result.success
        ):
            result = self._prune_candidate(
                node,
                raw_expression,
                result,
                X,
                y,
                node._elite_train_idx,
                node._elite_eval_idx,
            )
            node.sobolev_search_result = result
            node.sobolev_success = result.success
            node.sobolev_failure_type = result.failure_type.value
            node.sobolev_failure_message = result.failure_message
        self._set_sobolev_reward(node, result)

    def _set_sobolev_reward(self, node:Node, result) -> None:
        """Apply the existing SN penalty to one already base-evaluated candidate."""

        node.sobolev_penalty = (
            sobolev_penalty(result.term_novelties, self.sobolev_tau)
            if result is not None and result.success
            else 1.0
        )
        if self.structural_metric in {'sobolev', 'eic+sobolev'}:
            if self.sobolev_alpha == 0.0:
                node.reward = node.base_reward
            elif (result is None or not result.success) and self.sobolev_failure_policy == 'invalid':
                node.reward = 0.0
            else:
                node.reward = node.base_reward - self.sobolev_alpha * node.sobolev_penalty
        else:
            node.reward = node.base_reward
        node.final_reward = node.reward

    def _select_elite_candidate(
        self,
        pool:List[Node],
        X:Dict[str,np.ndarray],
        y:np.ndarray,
    ) -> Node:
        """Run lexicographic base screening, SN reranking and elite pruning."""

        if not pool:
            raise ValueError('SN-Elite received an empty simulation pool')
        base_order = rank_by_base(pool)
        shortlist_count = self.shortlist_config.count(len(base_order))
        shortlist = base_order[:shortlist_count]
        shortlist_ids = {id(candidate) for candidate in shortlist}
        self.sobolev_elite_iteration_count += 1
        self.sobolev_pool_candidate_count += len(pool)
        self.sobolev_shortlist_evaluation_count += len(shortlist)
        self.sobolev_base_rejected_count += len(pool) - len(shortlist)
        for rank, candidate in enumerate(base_order, start=1):
            candidate.base_rank = rank
            if id(candidate) in shortlist_ids:
                candidate.shortlist_status = 'shortlisted'
                candidate.selection_stage = 'sobolev_shortlist'
            else:
                candidate.shortlist_status = 'base_rejected'
                candidate.selection_stage = 'base_rejected'

        # Generation order is retained for evaluation so an evaluated rollout
        # parent can provide an incremental geometry key to its child.
        for candidate in pool:
            if id(candidate) in shortlist_ids:
                self._apply_deferred_sobolev(
                    candidate, X, y, allow_pruning=False,
                )

        pre_prune_order = rank_by_structural(shortlist)
        pre_prune_ranks = {
            id(candidate): rank
            for rank, candidate in enumerate(pre_prune_order, start=1)
        }
        if self.sobolev_pruning and self.sobolev_max_prunes > 0:
            prune_count = min(
                self.shortlist_config.prune_elite_k,
                len(pre_prune_order),
            )
            for candidate in pre_prune_order[:prune_count]:
                result = candidate.sobolev_search_result
                if result is not None and result.success:
                    raw_expression = '[' + ', '.join(
                        str(eqtree) for eqtree in candidate.eqtrees
                    ) + ']'
                    result = self._prune_candidate(
                        candidate,
                        raw_expression,
                        result,
                        X,
                        y,
                        candidate._elite_train_idx,
                        candidate._elite_eval_idx,
                    )
                    candidate.sobolev_search_result = result
                    candidate.sobolev_success = result.success
                    candidate.sobolev_failure_type = result.failure_type.value
                    candidate.sobolev_failure_message = result.failure_message
                    self._set_sobolev_reward(candidate, result)
                    self.sobolev_elite_prune_count += 1
                    candidate.selection_stage = 'elite_pruned'

        self.sobolev_penalties.extend(
            candidate.sobolev_penalty for candidate in shortlist
            if candidate.sobolev_penalty is not None
        )
        structural_order = rank_by_structural(shortlist)
        for rank, candidate in enumerate(structural_order, start=1):
            candidate.structural_rank = rank
            if candidate.selection_stage != 'elite_pruned':
                candidate.selection_stage = 'sobolev_reranked'
        selected = structural_order[0]
        selected.selection_stage = 'selected'
        selected.shortlist_status = 'selected'
        self._write_elite_iteration_trace(
            pool,
            shortlist_count,
            pre_prune_ranks,
            selected,
        )
        return selected

    def _write_elite_iteration_trace(
        self,
        pool:List[Node],
        shortlist_count:int,
        pre_prune_ranks:Dict[int, int],
        selected:Node,
    ) -> None:
        """Persist auditable pool membership and both deterministic rankings."""

        iteration_record = {
            'iteration': self.current_iter,
            'candidate_count': len(pool),
            'shortlist_count': shortlist_count,
            'shortlist_mode': self.shortlist_config.mode,
            'shortlist_size': self.shortlist_config.size,
            'shortlist_ratio': self.shortlist_config.ratio,
            'shortlist_min': self.shortlist_config.minimum,
            'shortlist_max': self.shortlist_config.maximum,
            'prune_elite_k': self.shortlist_config.prune_elite_k,
            'novelty_evaluated_count': sum(
                candidate.sobolev_evaluated for candidate in pool
            ),
            'selected_candidate_id': selected.candidate_id,
            'selected_expression': str(selected.phi),
            'selected_base_reward': self._finite_or_none(selected.base_reward),
            'selected_final_reward': self._finite_or_none(selected.final_reward),
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
        }
        self.sobolev_iteration_traces.append(iteration_record)
        if self.sobolev_iteration_log_path:
            self._append_jsonl(self.sobolev_iteration_log_path, iteration_record)
        if not self.sobolev_reranking_trace_path:
            return
        for candidate in rank_by_base(pool):
            self._append_jsonl(self.sobolev_reranking_trace_path, {
                'iteration': self.current_iter,
                'candidate_id': candidate.candidate_id,
                'raw_expression': '[' + ', '.join(
                    str(eqtree) for eqtree in candidate.eqtrees
                ) + ']',
                'fitted_expression': str(candidate.phi),
                'base_reward': self._finite_or_none(candidate.base_reward),
                'structural_reward': (
                    self._finite_or_none(candidate.final_reward)
                    if candidate.sobolev_evaluated else None
                ),
                'r2': self._finite_or_none(candidate.r2),
                'complexity': self._finite_or_none(candidate.complexity),
                'base_rank': candidate.base_rank,
                'pre_prune_structural_rank': pre_prune_ranks.get(id(candidate)),
                'structural_rank': candidate.structural_rank,
                'shortlist_status': candidate.shortlist_status,
                'selection_stage': candidate.selection_stage,
                'sobolev_evaluated': candidate.sobolev_evaluated,
                'sobolev_penalty': (
                    self._finite_or_none(candidate.sobolev_penalty)
                    if candidate.sobolev_evaluated else None
                ),
                'candidate_geometry_key': (
                    candidate.sobolev_search_result.candidate_geometry_key
                    if candidate.sobolev_search_result is not None else None
                ),
                'parent_geometry_key': (
                    candidate.sobolev_search_result.parent_geometry_key
                    if candidate.sobolev_search_result is not None else None
                ),
                'geometry_cache_hit': (
                    candidate.sobolev_search_result.geometry_cache_hit
                    if candidate.sobolev_search_result is not None else False
                ),
                'incremental_hit': (
                    candidate.sobolev_search_result.incremental_hit
                    if candidate.sobolev_search_result is not None else False
                ),
                'pruning_trace_id': candidate.pruning_trace_id,
                'selected': candidate is selected,
                'seed': self.random_state,
                'configuration_hash': self.configuration_hash,
            })

    def _sobolev_is_required(self) -> bool:
        return (
            self.structural_metric in {'sobolev', 'eic+sobolev'}
            or (self.sobolev_pruning and self.sobolev_max_prunes > 0)
        )

    def _evaluate_sobolev_expression(
        self,
        node:Node,
        raw_expression:str,
        fitted_expression,
        X:Dict[str,np.ndarray],
        coefficient_fitting_time:float = 0.0,
        eic_time:float = 0.0,
        parent_geometry_key:str|None = None,
        preparsed_hint:RefitGeometryHint|None = None,
    ):
        if parent_geometry_key is None:
            parent_geometry_key = node.sobolev_parent_geometry_key
        if (
            self.sobolev_evaluator_mode == 'incremental'
            and preparsed_hint is not None
        ):
            result = self.sobolev_evaluator.evaluate_preparsed(
                raw_expression=raw_expression,
                fitted_expression=fitted_expression,
                expression=preparsed_hint.expression,
                symbols=preparsed_hint.symbols,
                terms=preparsed_hint.terms,
                X=X,
                dataset_identity=self.sobolev_dataset_identity,
                geometry_indices=self.sobolev_geometry_indices,
                parent_geometry_key=parent_geometry_key,
            )
        else:
            result = self.sobolev_evaluator.evaluate(
                raw_expression=raw_expression,
                fitted_expression=fitted_expression,
                X=X,
                dataset_identity=self.sobolev_dataset_identity,
                geometry_indices=self.sobolev_geometry_indices,
                parent_geometry_key=(
                    parent_geometry_key
                    if self.sobolev_evaluator_mode == 'incremental'
                    else None
                ),
            )
        result.detailed_runtime.coefficient_fitting = coefficient_fitting_time
        result.detailed_runtime.eic = eic_time
        self.sobolev_evaluation_count += 1
        self.sobolev_total_evaluator_time += result.detailed_runtime.total_candidate_evaluation
        self.sobolev_full_parse_count += int(result.detailed_runtime.parsing > 0)
        self.sobolev_full_raw_geometry_count += int(
            not result.geometry_cache_hit and not result.incremental_hit
        )
        self.sobolev_geometry_cache_hit_count += int(result.geometry_cache_hit)
        self.sobolev_incremental_hit_count += int(result.incremental_hit)
        if result.incremental_fallback_reason:
            reason = result.incremental_fallback_reason
            self.sobolev_incremental_fallback_counts[reason] = (
                self.sobolev_incremental_fallback_counts.get(reason, 0) + 1
            )
        if result.success:
            self.sobolev_success_count += 1
        else:
            failure_name = result.failure_type.value
            self.sobolev_failure_counts[failure_name] = self.sobolev_failure_counts.get(failure_name, 0) + 1
            self._write_sobolev_failure(
                node, raw_expression, result, fitted_expression=str(fitted_expression),
            )
        algorithm_name = result.algorithm_used
        self.sobolev_algorithm_counts[algorithm_name] = self.sobolev_algorithm_counts.get(algorithm_name, 0) + 1
        return result

    def _prepare_additive_hint(
        self,
        coefficients:np.ndarray,
        eqtrees:List[nd.Symbol],
        feature_names:Tuple[str, ...],
    ) -> RefitGeometryHint|None:
        """Build the fitted additive SymPy AST from cached module ASTs."""

        if (
            self.sobolev_evaluator_mode != 'incremental'
            or not self._sobolev_is_required()
        ):
            return None
        try:
            symbols = tuple(sp.Symbol(name, real=True) for name in feature_names)
            components = []
            if float(coefficients[0]) != 0:
                components.append(sp.Float(str(float(coefficients[0]))))
            for coefficient, eqtree in zip(coefficients[1:], eqtrees, strict=True):
                if float(coefficient) == 0:
                    continue
                module_text = eqtree.to_str()
                cache_key = (module_text, feature_names)
                module_expression = self.sobolev_module_parse_cache.get(cache_key)
                if module_expression is None:
                    module_expression, module_symbols = parse_expression(
                        module_text, feature_names,
                    )
                    if tuple(module_symbols) != symbols:
                        raise ValueError('Module symbols do not match the fitted candidate')
                    self.sobolev_module_parse_cache[cache_key] = module_expression
                    self.sobolev_module_parse_misses += 1
                    while (
                        len(self.sobolev_module_parse_cache)
                        > self.sobolev_module_parse_cache_max_entries
                    ):
                        self.sobolev_module_parse_cache.popitem(last=False)
                else:
                    self.sobolev_module_parse_cache.move_to_end(cache_key)
                    self.sobolev_module_parse_hits += 1
                components.append(
                    sp.Mul(
                        sp.Float(str(float(coefficient))),
                        module_expression,
                        evaluate=False,
                    )
                )
            if not components:
                return None
            expression = sp.Add(*components, evaluate=False)
            terms = tuple(decompose_expand_mul(expression, symbols))
            return RefitGeometryHint(
                expression=expression,
                symbols=symbols,
                terms=terms,
                parent_geometry_key=None,
                removed_index=-1,
            )
        except Exception as error:
            self._write_pipeline_failure(Node(eqtrees), 'additive_preparsed_hint', error)
            return None

    def _prune_candidate(
        self,
        node:Node,
        raw_expression:str,
        initial_analysis,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
        train_idx:np.ndarray,
        eval_idx:np.ndarray,
    ):
        initial_fit = RefitResult(
            expression=node.phi,
            coefficients=list(initial_analysis.coefficients),
            r2=float(node.r2),
            complexity=int(node.complexity),
            eic=float(node.eic),
            base_reward=float(node.base_reward),
            success=True,
        )

        def refit_callback(current_fit, current_analysis, removed_index):
            return self._refit_after_deletion(
                current_fit, current_analysis, removed_index,
                X, y, train_idx, eval_idx,
            )

        def reevaluate_callback(refit):
            return self._evaluate_sobolev_expression(
                node, raw_expression, refit.expression, X,
                refit.coefficient_fitting_seconds, refit.eic_seconds,
                parent_geometry_key=(
                    refit.geometry_hint.parent_geometry_key
                    if refit.geometry_hint is not None
                    else initial_analysis.candidate_geometry_key
                ),
                preparsed_hint=(
                    refit.geometry_hint
                    if self.sobolev_pruning_geometry_reuse
                    else None
                ),
            )

        pruning = prune_and_refit(
            initial_fit,
            initial_analysis,
            PruningConfig(
                threshold=self.sobolev_tau,
                max_prunes=self.sobolev_max_prunes,
                acceptance_tolerance=self.sobolev_acceptance_tolerance,
            ),
            refit_callback,
            reevaluate_callback,
        )
        self.pruning_candidates_attempted += 1
        self.pruning_accepted_count += pruning.accepted_prunes
        self.pruning_rejected_count += pruning.rejected_prunes
        if pruning.accepted_prunes:
            self.pruning_candidates_changed += 1
            node.phi = pruning.final_fit.expression
            node.pruned_expression = str(node.phi)
            node.r2 = pruning.final_fit.r2
            node.complexity = pruning.final_fit.complexity
            node.eic = pruning.final_fit.eic
            node.base_reward = pruning.final_fit.base_reward
            node.reward = node.base_reward
        node.coefficient_fitting_time += sum(
            step.refit_coefficient_fitting_seconds for step in pruning.steps
        )
        node.eic_time += sum(step.refit_eic_seconds for step in pruning.steps)
        node.pruning_result = pruning
        node.pruning_trace_id = stable_state_id(
            f'{self.current_iter}|{self.candidate_counter}|{raw_expression}|{node.fitted_expression_before_pruning}'
        )
        termination = pruning.termination_reason
        self.pruning_termination_counts[termination] = self.pruning_termination_counts.get(termination, 0) + 1
        self._write_pruning_trace(node)
        return pruning.final_analysis

    def _refit_after_deletion(
        self,
        current_fit,
        current_analysis,
        removed_index:int,
        X:Dict[str,np.ndarray],
        y:np.ndarray,
        train_idx:np.ndarray,
        eval_idx:np.ndarray,
    ) -> RefitResult:
        fit_start = time.perf_counter()
        try:
            names = tuple(X.keys())
            expression, symbols = parse_expression(current_fit.expression, names)
            terms = decompose_expand_mul(expression, symbols)
            if len(terms) != len(current_analysis.terms):
                raise ValueError(
                    f'Refit term mismatch: parsed={len(terms)}, analysis={len(current_analysis.terms)}'
                )
            retained = [term for index, term in enumerate(terms) if index != removed_index]
            if not retained:
                raise ValueError('Pruning would remove every term')
            points = np.column_stack([np.asarray(X[name], dtype=float) for name in names])
            design = np.column_stack([
                evaluate_sympy(term.basis, symbols, points) for term in retained
            ])
            design[~np.isfinite(design)] = 0.0
            coefficients, _, _, _ = np.linalg.lstsq(
                design[train_idx, :], y[train_idx], rcond=None,
            )
            coefficients = np.round(coefficients, 6)
            prediction = design[eval_idx, :] @ coefficients
            r2 = R2_score(y[eval_idx], prediction)
            expression = sp.Add(*[
                sp.Mul(sp.Float(str(float(coefficient))), term.basis, evaluate=False)
                for coefficient, term in zip(coefficients, retained, strict=True)
                if coefficient != 0
            ], evaluate=False)
            if expression == 0:
                phi = nd.Number(0.0)
            else:
                phi = nd.parse(to_project_expression_string(expression))
            complexity = len(phi)
            coefficient_seconds = time.perf_counter() - fit_start
            eic_start = time.perf_counter()
            eic = get_eic(phi, X, random_state=self.eic_random_state)
            eic_seconds = time.perf_counter() - eic_start
            base_reward = self.get_reward(complexity, r2, eic)
            refitted_terms = tuple(
                TermSpec(float(coefficient), term.basis, term.canonical)
                for coefficient, term in zip(coefficients, retained, strict=True)
                if coefficient != 0
            )
            geometry_hint = (
                RefitGeometryHint(
                    expression=expression,
                    symbols=tuple(symbols),
                    terms=refitted_terms,
                    parent_geometry_key=current_analysis.candidate_geometry_key,
                    removed_index=removed_index,
                )
                if refitted_terms
                else None
            )
            return RefitResult(
                expression=phi,
                coefficients=coefficients.tolist(),
                r2=float(r2),
                complexity=int(complexity),
                eic=float(eic),
                base_reward=float(base_reward),
                success=True,
                coefficient_fitting_seconds=coefficient_seconds,
                eic_seconds=eic_seconds,
                geometry_hint=geometry_hint,
            )
        except Exception as error:
            return RefitResult(
                success=False,
                failure_type=type(error).__name__,
                failure_message=str(error),
                coefficient_fitting_seconds=time.perf_counter() - fit_start,
            )

    def _write_candidate(self, node:Node) -> None:
        if not self.sobolev_detailed_logging or not self.sobolev_candidate_log_path:
            return
        raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
        result = node.sobolev_search_result
        record = {
            'candidate_id': self.candidate_counter,
            'parent_state_id': stable_state_id(str(node.parent.eqtrees)) if node.parent else None,
            'state_id': stable_state_id(raw_expression),
            'iteration': self.current_iter,
            'fit_type': node.fit_type,
            'raw_expression': raw_expression,
            'fitted_expression': node.fitted_expression_before_pruning or str(node.phi),
            'pruned_expression': node.pruned_expression,
            'coefficients': result.coefficients if result else [],
            'train_r2': self._finite_or_none(node.r2),
            'test_r2': self._finite_or_none(node.r2) if self.ratio == 1.0 else None,
            'nmse': self._finite_or_none(1 - node.r2) if self.ratio == 1.0 else None,
            'complexity': self._finite_or_none(node.complexity),
            'term_count': len(result.terms) if result else None,
            'eic': self._finite_or_none(node.eic),
            'terms': result.terms if result else [],
            'term_novelties': result.term_novelties if result else [],
            'min_novelty': result.min_novelty if result else None,
            'mean_novelty': result.mean_novelty if result else None,
            'low_novelty_ratio': result.low_novelty_ratio if result else None,
            'sobolev_penalty': node.sobolev_penalty,
            'sobolev_evaluated': node.sobolev_evaluated,
            'selection_stage': node.selection_stage,
            'shortlist_status': node.shortlist_status,
            'base_rank': node.base_rank,
            'structural_rank': node.structural_rank,
            'base_reward_before_pruning': self._finite_or_none(node.base_reward_before_pruning),
            'base_reward': self._finite_or_none(node.base_reward),
            'final_reward': self._finite_or_none(node.final_reward),
            'pruning_trace_id': node.pruning_trace_id,
            'pruning_accepted_count': (
                node.pruning_result.accepted_prunes if node.pruning_result else 0
            ),
            'pruning_rejected_count': (
                node.pruning_result.rejected_prunes if node.pruning_result else 0
            ),
            'pruning_termination_reason': (
                node.pruning_result.termination_reason if node.pruning_result else None
            ),
            'novelty_search_subset': True,
            'novelty_full_recomputed': False,
            'evaluator_runtime': result.detailed_runtime.total_candidate_evaluation if result else 0.0,
            'coefficient_fitting_runtime': node.coefficient_fitting_time,
            'eic_runtime': node.eic_time,
            'candidate_runtime': node.candidate_runtime,
            'cache_hits': result.cache_hits if result else 0,
            'cache_misses': result.cache_misses if result else 0,
            'algorithm_used': result.algorithm_used if result else None,
            'fallback_reason': result.fallback_reason if result else None,
            'candidate_geometry_key': result.candidate_geometry_key if result else None,
            'parent_geometry_key': result.parent_geometry_key if result else None,
            'geometry_cache_hit': result.geometry_cache_hit if result else False,
            'incremental_hit': result.incremental_hit if result else False,
            'incremental_fallback_reason': result.incremental_fallback_reason if result else None,
            'geometry_reuse_mode': result.geometry_reuse_mode if result else None,
            'geometry_fallback_reason': result.geometry_fallback_reason if result else None,
            'reused_term_count': result.reused_term_count if result else 0,
            'added_term_count': result.added_term_count if result else 0,
            'removed_term_count': result.removed_term_count if result else 0,
            'changed_term_count': result.changed_term_count if result else 0,
            'success': result.success if result else True,
            'failure_type': result.failure_type.value if result else None,
            'failure_message': result.failure_message if result else None,
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
        }
        self._append_jsonl(self.sobolev_candidate_log_path, record)

    def _write_sobolev_failure(
        self, node:Node, raw_expression:str, result, fitted_expression:str|None = None,
    ) -> None:
        if not self.sobolev_failure_log_path:
            return
        self._append_jsonl(self.sobolev_failure_log_path, {
            'iteration': self.current_iter,
            'raw_expression': raw_expression,
            'fitted_expression': fitted_expression or str(node.phi),
            'stage': 'sobolev_candidate_evaluation',
            'failure_type': result.failure_type.value,
            'exception': result.failure_message,
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
        })

    def _write_pruning_trace(self, node:Node) -> None:
        if not self.sobolev_pruning_trace_path or node.pruning_result is None:
            return
        self._append_jsonl(self.sobolev_pruning_trace_path, {
            'trace_id': node.pruning_trace_id,
            'candidate_iteration': self.current_iter,
            'candidate_variant_index': self.candidate_counter + 1,
            'raw_expression': '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']',
            'fit_type': node.fit_type,
            'pruning': node.pruning_result.as_dict(),
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
        })

    def _write_pipeline_failure(self, node:Node, stage:str, error:Exception) -> None:
        key = f'{stage}:{type(error).__name__}'
        self.pipeline_failure_counts[key] = self.pipeline_failure_counts.get(key, 0) + 1
        if not self.sobolev_detailed_logging or not self.sobolev_failure_log_path:
            return
        raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
        self._append_jsonl(self.sobolev_failure_log_path, {
            'iteration': self.current_iter,
            'raw_expression': raw_expression,
            'fitted_expression': str(node.phi) if node.phi is not None else None,
            'stage': stage,
            'failure_type': type(error).__name__,
            'exception': str(error),
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
        })

    def _finalize_run(self, X:Dict[str,np.ndarray]) -> None:
        self._recompute_sobolev_elites_full(X)
        if not self.sobolev_run_summary_path:
            return
        penalties = np.asarray(self.sobolev_penalties, dtype=float)
        cache = self.sobolev_evaluator.cache if self.sobolev_evaluator is not None else None
        record = {
            'structural_metric': self.structural_metric,
            'sobolev_pruning': self.sobolev_pruning,
            'sobolev_max_prunes': self.sobolev_max_prunes,
            'sobolev_evaluator_mode': self.sobolev_evaluator_mode,
            'sobolev_pruning_geometry_reuse': self.sobolev_pruning_geometry_reuse,
            'sobolev_selection_mode': self.sobolev_selection_mode,
            'shortlist_mode': self.shortlist_config.mode,
            'shortlist_size': self.shortlist_config.size,
            'shortlist_ratio': self.shortlist_config.ratio,
            'shortlist_min': self.shortlist_config.minimum,
            'shortlist_max': self.shortlist_config.maximum,
            'prune_elite_k': self.shortlist_config.prune_elite_k,
            'seed': self.random_state,
            'configuration_hash': self.configuration_hash,
            'candidate_variants_evaluated': self.candidate_counter,
            'unique_raw_expressions': len(self.unique_raw_expressions),
            'unique_fitted_expressions': len(self.unique_fitted_expressions),
            'sobolev_evaluations': self.sobolev_evaluation_count,
            'sobolev_success_count': self.sobolev_success_count,
            'sobolev_success_rate': (
                self.sobolev_success_count / self.sobolev_evaluation_count
                if self.sobolev_evaluation_count else None
            ),
            'sobolev_failure_counts': self.sobolev_failure_counts,
            'pipeline_failure_counts': self.pipeline_failure_counts,
            'pruning_candidates_attempted': self.pruning_candidates_attempted,
            'pruning_candidates_changed': self.pruning_candidates_changed,
            'pruning_accepted_count': self.pruning_accepted_count,
            'pruning_rejected_count': self.pruning_rejected_count,
            'pruning_termination_counts': self.pruning_termination_counts,
            'penalty_nonzero_count': int(np.sum(penalties > 0)) if penalties.size else 0,
            'penalty_mean': float(np.mean(penalties)) if penalties.size else None,
            'penalty_median': float(np.median(penalties)) if penalties.size else None,
            'penalty_max': float(np.max(penalties)) if penalties.size else None,
            'algorithm_counts': self.sobolev_algorithm_counts,
            'gram_fast_path_rate': (
                self.sobolev_algorithm_counts.get('gram_cholesky', 0) / self.sobolev_evaluation_count
                if self.sobolev_evaluation_count else None
            ),
            'evaluator_total_seconds': self.sobolev_total_evaluator_time,
            'geometry_cache_hit_count': self.sobolev_geometry_cache_hit_count,
            'incremental_hit_count': self.sobolev_incremental_hit_count,
            'incremental_fallback_counts': self.sobolev_incremental_fallback_counts,
            'full_parse_count': self.sobolev_full_parse_count,
            'full_raw_geometry_count': self.sobolev_full_raw_geometry_count,
            'module_parse_hits': self.sobolev_module_parse_hits,
            'module_parse_misses': self.sobolev_module_parse_misses,
            'module_parse_cache_entries': len(self.sobolev_module_parse_cache),
            'base_fit_eic_total_seconds': self.base_fit_eic_total_seconds,
            'pool_candidate_count': self.sobolev_pool_candidate_count,
            'shortlist_evaluation_count': self.sobolev_shortlist_evaluation_count,
            'base_rejected_count': self.sobolev_base_rejected_count,
            'elite_prune_count': self.sobolev_elite_prune_count,
            'elite_iteration_count': self.sobolev_elite_iteration_count,
            'search_total_seconds': time.time() - self.start_time,
            'cache_hits': cache.hits if cache else 0,
            'cache_misses': cache.misses if cache else 0,
            'cache_hit_rate': cache.hits / (cache.hits + cache.misses) if cache and (cache.hits + cache.misses) else None,
            'cache_memory_bytes': cache.memory_bytes if cache else 0,
        }
        geometry_cache = (
            self.sobolev_evaluator.geometry_cache
            if self.sobolev_evaluator is not None
            else None
        )
        record.update({
            'geometry_cache_hits': geometry_cache.hits if geometry_cache else 0,
            'geometry_cache_misses': geometry_cache.misses if geometry_cache else 0,
            'geometry_cache_failure_hits': geometry_cache.failure_hits if geometry_cache else 0,
            'geometry_cache_entries': geometry_cache.entry_count if geometry_cache else 0,
            'geometry_cache_memory_bytes': geometry_cache.memory_bytes if geometry_cache else 0,
            'geometry_cache_evictions': geometry_cache.evictions if geometry_cache else 0,
            'incremental_lookups': geometry_cache.incremental_lookups if geometry_cache else 0,
            'incremental_parent_hits': geometry_cache.incremental_hits if geometry_cache else 0,
        })
        path = Path(self.sobolev_run_summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    def _recompute_sobolev_elites_full(self, X:Dict[str,np.ndarray]) -> None:
        if self.sobolev_evaluator is None or self.best is None:
            return
        full_config = replace(self.sobolev_evaluator.config, geometry_sample_size=None)
        evaluator = SobolevEvaluator(full_config, TermEvaluationCache(enabled=self.sobolev_cache))
        elites = [('best', self.best), *[(f'pareto_{index}', node) for index, node in enumerate(self.pareto_front)]]
        full_indices = np.arange(len(next(iter(X.values()))), dtype=np.int64)
        for label, node in elites:
            raw_expression = '[' + ', '.join(str(eqtree) for eqtree in node.eqtrees) + ']'
            result = evaluator.evaluate(
                raw_expression,
                node.phi,
                X,
                dataset_identity=self.sobolev_dataset_identity,
                geometry_indices=full_indices,
            )
            node.sobolev_full_result = result
            if self.sobolev_elite_log_path:
                self._append_jsonl(self.sobolev_elite_log_path, {
                    'elite': label,
                    'raw_expression': raw_expression,
                    'fitted_expression': str(node.phi),
                    'novelty_search_subset': False,
                    'novelty_full_recomputed': True,
                    'result': result.as_dict(),
                    'seed': self.random_state,
                    'configuration_hash': self.configuration_hash,
                })

    @staticmethod
    def _append_jsonl(path_value, record) -> None:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def _finite_or_none(value):
        if value is None:
            return None
        return float(value) if np.isfinite(value) else None
