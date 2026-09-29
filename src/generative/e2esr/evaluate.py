# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# licenses/e2esr.txt file in the root directory of this repository.
# Modified for repository integration: license location and portable imports.

import copy
import json

from pathlib import Path

from logging import getLogger
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from copy import deepcopy
from .symbolicregression.utils import to_cuda
import glob
import scipy

from .parsers import get_parser
from . import symbolicregression
from .symbolicregression.slurm import init_signal_handler, init_distributed_mode
from .symbolicregression.utils import bool_flag, initialize_exp
from .symbolicregression.model import check_model_params, build_modules
from .symbolicregression.envs import build_env
from .symbolicregression.trainer import Trainer
from .symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor
from .symbolicregression.model.model_wrapper import ModelWrapper
from .symbolicregression.metrics import compute_metrics
from sklearn.model_selection import train_test_split
import pandas as pd

from tqdm import tqdm
import time

# np.seterr(all="raise")


def read_file(filename, label="target", sep=None):
    filename = str(filename)

    if filename.endswith("gz"):
        compression = "gzip"
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(
            filename, sep=sep, compression=compression, engine="python"
        )

    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert X.shape[1] == feature_names.shape[0]

    return X, y, feature_names


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        Evaluator.ENV = trainer.env

    def set_env_copies(self, data_types):
        for data_type in data_types:
            setattr(self, "{}_env".format(data_type), deepcopy(self.env))

    def evaluate_pmlb(
        self,
        target_noise=0.0,
        random_state=29910,
        verbose=False,
        save=True,
        filter_fn=None,
        logger=None,
        save_file=None,
        save_suffix="eval_pmlb.jsonl",
    ):
        scores = defaultdict(list)
        env = self.env
        params = self.params
        embedder = (
            self.modules["embedder"].module
            if params.multi_gpu
            else self.modules["embedder"]
        )

        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        embedder.eval()
        encoder.eval()
        decoder.eval()

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            max_number_bags=params.max_number_bags,
            rescale=params.rescale,
        )

        metadata_path = Path(params.pmlb_metadata_path)
        all_datasets = pd.read_csv(
            metadata_path / "all_summary_stats.tsv",
            sep="\t",
        )
        regression_datasets = all_datasets[all_datasets["task"] == "regression"]
        regression_datasets = regression_datasets[
            regression_datasets["n_categorical_features"] == 0
        ]
        problems = regression_datasets

        if filter_fn is not None:
            problems = problems[filter_fn(problems)]
        problem_names = problems["dataset"].values.tolist()
        pmlb_path = Path(params.pmlb_data_path)

        # feynman_problems = feynman_problems[["Filename", "Formula"]].dropna().values
        # feynman_formulas = {}
        # for p in range(feynman_problems.shape[0]):
        #     feynman_formulas[
        #         "feynman_" + feynman_problems[p][0].replace(".", "_")
        #     ] = feynman_problems[p][1]
        feynman_formulas = pd.read_csv(
            metadata_path / "feynman_equations.tsv", sep="\t"
        ).set_index("Filename")["Formula"].to_dict()

        first_write = 0
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path
                    if self.params.eval_dump_path is not None
                    else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = os.path.join(save_file, save_suffix)

        rng = np.random.RandomState(random_state)
        pbar = tqdm(total=len(problem_names))
        for problem_name in problem_names:
            if problem_name in feynman_formulas:
                formula = feynman_formulas[problem_name]
            else:
                formula = "???"

            X, y, _ = read_file(
                pmlb_path / problem_name / f"{problem_name}.tsv.gz"
            )
            y = np.expand_dims(y, -1)

            x_to_fit, x_to_predict, y_to_fit, y_to_predict = train_test_split(
                X, y, test_size=0.25, shuffle=True, random_state=random_state
            )

            scale = target_noise * np.sqrt(np.mean(np.square(y_to_fit)))
            noise = rng.normal(loc=0.0, scale=scale, size=y_to_fit.shape)
            y_to_fit += noise

            try:
                dstr.fit(x_to_fit, y_to_fit, verbose=verbose)
                problem_results = defaultdict(list)
            except KeyboardInterrupt:
                print(f"Keyboard interrupt, skip problem {problem_name}")
                continue
           
            for refinement_type in dstr.retrieve_refinements_types():
                best_gen = copy.deepcopy(
                    dstr.retrieve_tree(refinement_type=refinement_type, with_infos=True)
                )
                predicted_tree = best_gen["predicted_tree"]
                if predicted_tree is None:
                    continue
                del best_gen["predicted_tree"]
                if "metrics" in best_gen:
                    del best_gen["metrics"]

                problem_results["predicted_tree"].append(predicted_tree)
                problem_results["predicted_tree_prefix"].append(
                    predicted_tree.prefix() if predicted_tree is not None else None
                )
                for info, val in best_gen.items():
                    problem_results[info].append(val)

                y_tilde_to_fit = dstr.predict(x_to_fit, refinement_type=refinement_type)
                results_fit = compute_metrics(
                    {
                        "true": [y_to_fit],
                        "predicted": [y_tilde_to_fit],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    problem_results[k + "_fit"].extend(v)
                    scores[refinement_type + "|" + k + "_fit"].extend(v)

                y_tilde_to_predict = dstr.predict(
                    x_to_predict, refinement_type=refinement_type
                )
                results_predict = compute_metrics(
                    {
                        "true": [y_to_predict],
                        "predicted": [y_tilde_to_predict],
                        "predicted_tree": [predicted_tree],
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_predict.items():
                    problem_results[k + "_predict"].extend(v)
                    scores[refinement_type + "|" + k + "_predict"].extend(v)

            problem_results = pd.DataFrame.from_dict(problem_results)
            problem_results.insert(0, "problem", problem_name)
            problem_results.insert(0, "formula", formula)
            problem_results["input_dimension"] = x_to_fit.shape[1]
            problem_results['epoch'] = self.trainer.epoch

            if save:
                with open(save_file, "a") as f:
                    for idx, line in problem_results.iterrows():
                        line = line.to_dict()
                        for k, v in line.items():
                            if isinstance(v, np.float32) or isinstance(v, np.float64):
                                line[k] = float(v)
                            elif isinstance(v, np.integer):
                                line[k] = int(v)
                            else:
                                line[k] = str(v)
                        f.write(json.dumps(line) + "\n")
            pbar.update(1)
        for k, v in scores.items():
            scores[k] = np.nanmean(v)
        return scores

    def evaluate_in_domain(
        self,
        data_type,
        task,
        verbose=True,
        ablation_to_keep=None,
        save=False,
        logger=None,
        save_file=None,
    ):

        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        scores = OrderedDict({"epoch": self.trainer.epoch})

        params = self.params
        logger.info(
            "====== STARTING EVALUATION (multi-gpu: {}) =======".format(
                params.multi_gpu
            )
        )

        embedder = (
            self.modules["embedder"].module
            if params.multi_gpu
            else self.modules["embedder"]
        )

        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        embedder.eval()
        encoder.eval()
        decoder.eval()

        env = getattr(self, "{}_env".format(data_type))

        eval_size_per_gpu = params.eval_size
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=eval_size_per_gpu,
            input_length_modulo=params.eval_input_length_modulo,
            test_env_seed=self.params.test_env_seed,
        )

        mw = ModelWrapper(
            env=env,
            embedder=embedder,
            encoder=encoder,
            decoder=decoder,
            beam_length_penalty=params.beam_length_penalty,
            beam_size=params.beam_size,
            max_generated_output_len=params.max_generated_output_len,
            beam_early_stopping=params.beam_early_stopping,
            beam_temperature=params.beam_temperature,
            beam_type=params.beam_type,
        )

        dstr = SymbolicTransformerRegressor(
            model=mw,
            max_input_points=params.max_input_points,
            n_trees_to_refine=params.n_trees_to_refine,
            rescale=False,
        )

        first_write = True
        if save:
            if save_file is None:
                save_file = (
                    self.params.eval_dump_path
                    if self.params.eval_dump_path is not None
                    else self.params.dump_path
                )
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = os.path.join(save_file, "eval_in_domain.jsonl")

        batch_before_writing_threshold = min(
            2, eval_size_per_gpu // params.batch_size_eval
        )
        batch_before_writing = batch_before_writing_threshold

        if ablation_to_keep is not None:
            ablation_to_keep = list(
                map(lambda x: "info_" + x, ablation_to_keep.split(","))
            )
        else:
            ablation_to_keep = []

        pbar = tqdm(total=eval_size_per_gpu)

        batch_results = defaultdict(list)

        for samples, _ in iterator:
            x_to_fit = samples["x_to_fit"]
            y_to_fit = samples["y_to_fit"]
            infos = samples["infos"]
            tree = samples["tree"]

            dstr.fit(x_to_fit, y_to_fit, verbose=verbose)

            # dstr.tree = [[{"predicted_tree": tree_i, "refinement_type": ref} for ref in dstr.retrieve_refinements_types()] for tree_i in tree] ##TO DEBUG METRICS
            # dstr.beam_selection_metrics=0
            for k, v in infos.items():
                infos[k] = v.tolist()

            for refinement_type in dstr.retrieve_refinements_types():

                best_gens = copy.deepcopy(
                    dstr.retrieve_tree(
                        refinement_type=refinement_type, dataset_idx=-1, with_infos=True
                    )
                )
                predicted_tree = [best_gen["predicted_tree"] for best_gen in best_gens]
                for best_gen in best_gens:
                    del best_gen["predicted_tree"]
                    if "metrics" in best_gen:
                        del best_gen["metrics"]

                batch_results["predicted_tree"].extend(predicted_tree)
                batch_results["predicted_tree_prefix"].extend(
                    [
                        _tree.prefix() if _tree is not None else np.nan
                        for _tree in predicted_tree
                    ]
                )
                for best_gen in best_gens:
                    for info, val in best_gen.items():
                        batch_results[info].extend([val])

                for k, v in infos.items():
                    batch_results["info_" + k].extend(v)

                y_tilde_to_fit = dstr.predict(
                    x_to_fit, refinement_type=refinement_type, batch=True
                )
                assert len(y_to_fit) == len(
                    y_tilde_to_fit
                ), "issue with len, tree: {}, x:{} true: {}, predicted: {}".format(
                    len(predicted_tree),
                    len(x_to_fit),
                    len(y_to_fit),
                    len(y_tilde_to_fit),
                )
                results_fit = compute_metrics(
                    {
                        "true": y_to_fit,
                        "predicted": y_tilde_to_fit,
                        "tree": tree,
                        "predicted_tree": predicted_tree,
                    },
                    metrics=params.validation_metrics,
                )
                for k, v in results_fit.items():
                    batch_results[k + "_fit"].extend(v)
                del results_fit

                if self.params.prediction_sigmas is None:
                    prediction_sigmas = []
                else:
                    prediction_sigmas = [
                        float(sigma)
                        for sigma in self.params.prediction_sigmas.split(",")
                    ]
                for sigma in prediction_sigmas:
                    x_to_predict = samples["x_to_predict_{}".format(sigma)]
                    y_to_predict = samples["y_to_predict_{}".format(sigma)]
                    y_tilde_to_predict = dstr.predict(
                        x_to_predict, refinement_type=refinement_type, batch=True
                    )
                    results_predict = compute_metrics(
                        {
                            "true": y_to_predict,
                            "predicted": y_tilde_to_predict,
                            "tree": tree,
                            "predicted_tree": predicted_tree,
                        },
                        metrics=params.validation_metrics,
                    )
                    for k, v in results_predict.items():
                        batch_results[k + "_predict_{}".format(sigma)].extend(v)
                    del results_predict

                batch_results["tree"].extend(tree)
                batch_results["tree_prefix"].extend([_tree.prefix() for _tree in tree])
                
            if save:

                batch_before_writing -= 1
                if batch_before_writing <= 0:
                    batch_results = pd.DataFrame.from_dict(batch_results)
                    batch_results['epoch'] = self.trainer.epoch
                    if first_write:
                        with open(save_file, "a") as f:
                            for idx, line in batch_results.iterrows():
                                line = line.to_dict()
                                for k, v in line.items():
                                    if isinstance(v, np.float32) or isinstance(v, np.float64):
                                        line[k] = float(v)
                                    elif isinstance(v, np.integer):
                                        line[k] = int(v)
                                    else:
                                        line[k] = str(v)
                                f.write(json.dumps(line) + "\n")

                        if logger is not None:
                            logger.info("Just started saving")
                        first_write = False
                    else:
                        with open(save_file, "a") as f:
                            for idx, line in batch_results.iterrows():
                                line = line.to_dict()
                                for k, v in line.items():
                                    if isinstance(v, np.float32) or isinstance(v, np.float64):
                                        line[k] = float(v)
                                    elif isinstance(v, np.integer):
                                        line[k] = int(v)
                                    else:
                                        line[k] = str(v)
                                f.write(json.dumps(line) + "\n")
                        if logger is not None:
                            logger.info(
                                "Saved {} equations".format(
                                    self.params.batch_size_eval
                                    * batch_before_writing_threshold
                                )
                            )
                    batch_before_writing = batch_before_writing_threshold
                    batch_results = defaultdict(list)

            bs = len(x_to_fit)
            pbar.update(bs)

        try:
            # df = pd.read_csv(save_file, na_filter=True)
            df = pd.read_json(save_file, lines=True)
            df = df[df['epoch'] == self.trainer.epoch]
            df = df.drop(columns=['epoch'])
        except:
            logger.info("WARNING: no results")
            return
        info_columns = filter(lambda x: x.startswith("info_"), df.columns)
        df = df.drop(columns=filter(lambda x: x not in ablation_to_keep, info_columns))

        for refinement_type, df_refinement_type in df.groupby("refinement_type"):
            obj_cols = df.select_dtypes(include=['object']).columns
            df_refinement_type = df_refinement_type.drop(columns=obj_cols)
            avg_scores = df_refinement_type.mean().to_dict()
            for k, v in avg_scores.items():
                scores[refinement_type + "|" + k] = v
            # avg_scores_failure = df_refinement_type.isna().mean()
            # for k, v in avg_scores_failure.items(): scores[refinement_type + "|" + k + "_false"]=v
            for ablation in ablation_to_keep:
                for val, df_ablation in df_refinement_type.groupby(ablation):
                    avg_scores_ablation = df_ablation.mean()
                    for k, v in avg_scores_ablation.items():
                        scores[
                            refinement_type + "|" + k + "_{}_{}".format(ablation, val)
                        ] = v
                    # avg_scores_ablation_failure = df_ablation.isna().mean()
                    # for k, v in avg_scores_ablation_failure.items(): scores[refinement_type + "|" + k + "_{}_{}_false".format(ablation, val)]=v
        return scores


def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params, write_dump_path=False)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if not params.cpu:
        assert torch.cuda.is_available()
    params.eval_only = True
    symbolicregression.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    if params.batch_size_eval is None:
        params.batch_size_eval = int(1.5 * params.batch_size)

    env = build_env(params)
    env.rng = np.random.RandomState(0)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    scores = {}
    save = params.save_results

    if params.eval_in_domain:
        evaluator.set_env_copies(["valid1"])
        scores = evaluator.evaluate_in_domain(
            "valid1",
            "functions",
            save=save,
            logger=logger,
            ablation_to_keep=params.ablation_to_keep,
        )
        logger.info("__log__:%s" % json.dumps(scores))

    if params.eval_on_pmlb:
        target_noise = params.target_noise
        random_state = params.random_state
        data_type = params.pmlb_data_type

        if data_type == "feynman":
            filter_fn = lambda x: x["dataset"].str.contains("feynman")
        elif data_type == "strogatz":
            print("Strogatz data")
            filter_fn = lambda x: x["dataset"].str.contains("strogatz")
        elif data_type == "603_fri_c0_250_50":
            filter_fn = lambda x: x["dataset"].str.contains("603_fri_c0_250_50")
        else:
            filter_fn = lambda x: ~(
                x["dataset"].str.contains("strogatz")
                | x["dataset"].str.contains("feynman")
            )

        pmlb_scores = evaluator.evaluate_pmlb(
            target_noise=target_noise,
            verbose=params.eval_verbose_print,
            random_state=random_state,
            save=save,
            filter_fn=filter_fn,
            logger=logger,
            save_file=None,
            save_suffix="eval_pmlb.jsonl",
        )
        logger.info("__pmlb__:%s" % json.dumps(pmlb_scores))


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params)
