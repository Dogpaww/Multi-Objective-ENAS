import random

import numpy as np
import torch
import json
import random
import os
from copy import deepcopy

import gc
import torch

import numpy as np
import math
import random
import torch
import torchvision
import csv
import hashlib
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from medmnist import INFO
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from numpy import savetxt
from datetime import datetime
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt

from evaluate import Evaluate
from mealpy import Tuner
from mealpy.evolutionary_based.SHADE import L_SHADE
from mealpy.evolutionary_based.DE import L_SHADE, BaseDE
from mealpy.evolutionary_based.ES import CMA_ES
from mealpy.evolutionary_based.GA import BaseGA
from mealpy.swarm_based import PSO
from mealpy.swarm_based.ACOR import OriginalACOR
from mealpy.utils import io
from model import NetworkCIFAR
import operations_mapping
from utils import decode_cell, decode_operations
from optimizer import Optimizer

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import IntegerRandomSampling,FloatRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.problem import ElementwiseProblem #added imports
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting #added imports

def evaluate_arch(self, ind, dataset, measure):

    return random.randint(10,10)

class NASNSGA2Problem(ElementwiseProblem): #added NSGA wrapper class to use x is one candidate arch vector,out["f"]=two obj vector 
    def __init__(self, soga, n_var=48, proxy_name="synflow"):
        super().__init__(
            n_var=n_var,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.zeros(n_var),
            xu=np.ones(n_var) * 0.99,
        )
        self.soga = soga
        self.proxy_name = proxy_name

    """def _evaluate(self, x, out, *args, **kwargs):
        try:
            record = self.soga.evaluate_architecture_metrics(
                x,
                proxy_name=self.proxy_name
            )

            # pymoo minimizes all objectives.
            # Objective 1: maximize proxy score -> minimize negative log proxy.
            # Objective 2: minimize FLOPs.
            proxy_obj = -record["log_proxy_score"]
            flops_obj = record["flops_billion"]
            zico_obj=-record["zico"]

            out["F"] = np.array([proxy_obj, flops_obj,zico_obj], dtype=float)

        except Exception as e:
            record = {
                "solution": [float(v) for v in x],
                "error": repr(e),
                "proxy_score": 0.0,
                "log_proxy_score": -300.0,
                "flops": float("inf"),
                "flops_billion": float("inf"),
            }

            # Bad architecture gets punished.
            out["F"] = np.array([1e9, 1e9, 1e9], dtype=float)"""
    
    def _evaluate(self, x, out, *args, **kwargs):

        try:
            record = self.soga.evaluate_architecture_metrics(
                x,
                proxy_name=self.proxy_name
            )

            proxy_obj = -record["log_proxy_score"]
            flops_obj = record["flops_billion"]
            zico_obj = -record["zico"]

            print("DEBUG OBJECTIVES:", proxy_obj, flops_obj, zico_obj)

            out["F"] = np.array([proxy_obj, flops_obj, zico_obj], dtype=float)

            record["objective_vector"] = [
            float(proxy_obj),
            float(flops_obj),
            float(zico_obj)
            ]               

            self.soga.nsga2_archive.append(record)

        except Exception as e:
            print("\n================ NSGA-II EVALUATION ERROR ================")
            print("Candidate x:", x)
            import traceback
            traceback.print_exc()
            print("==========================================================\n")

            raise



class SOGA(Optimizer): 

    

    def build_model_from_individual(self, individual, is_final=False):
        """
        Rebuild a NetworkCIFAR model from the exact saved NSGA-II individual.

        Use this for final training because record["individual"] is the actual
        decoded architecture selected from the Pareto front.
        """
        info = INFO[self.medmnist_dataset]
        n_classes = len(info["label"])

        individual = list(individual)

        # JSON may store integers as floats, so force operation choices/layer count back to int where needed.
        fixed_individual = []
        for i, v in enumerate(individual):
            if isinstance(v, float) and float(v).is_integer():
                fixed_individual.append(int(v))
            else:
                fixed_individual.append(v)

        fixed_individual[-1] = int(fixed_individual[-1])

        decoded_cell = decode_cell(
            decode_operations(fixed_individual[:-1], self.pop.indexes)
        )

        model = NetworkCIFAR(
            self.n_channels,
            n_classes,
            fixed_individual[-1],
            True,
            decoded_cell,
            self.is_medmnist,
            is_final,
            self.dropout_rate,
            "FP32",
            False
        )

        return model, decoded_cell, n_classes

    def build_model_from_solution(self, solution):
        info = INFO[self.medmnist_dataset]
        n_classes = len(info['label'])

        solution = np.asarray(solution, dtype=float)
        solution = np.clip(solution, 0.0, 0.99)

        individual = []

        for i in range(32):
            if i % 2 == 0:
                individual.append(float(solution[i]))
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                individual.append(int(math.floor(solution[i] * len(self.attentions))))

        num_layers = int(math.floor(2 + ((self.layers - 2) * solution[-1])))
        num_layers = max(2, min(int(self.layers), num_layers))
        individual.append(num_layers)

        decoded_cell = decode_cell(
            decode_operations(individual[:-1], self.pop.indexes)
        )

        is_final = False

        decoded_model = NetworkCIFAR(
            self.n_channels,
            n_classes,
            individual[-1],
            True,
            decoded_cell,
            self.is_medmnist,
            is_final,
            self.dropout_rate,
            'FP32',
            False
        )

        return individual, decoded_cell, decoded_model, n_classes
    
    def rank_architectures_topsis(
        self,
        records,
        weights=(0.4, 0.4, 0.2),
        criteria=("log_proxy_score", "zico", "flops_billion"),
        benefit=(True, True, False),
    ):
        """
        TOPSIS ranking for Pareto-front architecture selection.

        Criteria:
            log_proxy_score: benefit criterion, higher is better
            zico: benefit criterion, higher is better
            flops_billion: cost criterion, lower is better

        Weights:
            log_proxy_score = 0.4
            zico = 0.4
            flops_billion = 0.2
        """

        eps = 1e-12

        valid = []
        for r in records:
            ok = True
            for c in criteria:
                if c not in r:
                    ok = False
                    break
                if not np.isfinite(float(r[c])):
                    ok = False
                    break
            if ok:
                valid.append(r)

        if len(valid) == 0:
            raise RuntimeError("No valid architectures available for TOPSIS ranking.")

        # Decision matrix
        X = np.array(
            [[float(r[c]) for c in criteria] for r in valid],
            dtype=float
        )

        # Normalize weights
        weights = np.array(weights, dtype=float)
        weights = weights / (np.sum(weights) + eps)

        # Step 1: vector normalization
        norm = np.sqrt(np.sum(X ** 2, axis=0)) + eps
        R = X / norm

        # Step 2: weighted normalized decision matrix
        V = R * weights

        # Step 3: ideal best and ideal worst
        ideal_best = np.zeros(len(criteria))
        ideal_worst = np.zeros(len(criteria))

        for j, is_benefit in enumerate(benefit):
            if is_benefit:
                ideal_best[j] = np.max(V[:, j])
                ideal_worst[j] = np.min(V[:, j])
            else:
                ideal_best[j] = np.min(V[:, j])
                ideal_worst[j] = np.max(V[:, j])

        # Step 4: distance from ideal best and ideal worst
        d_best = np.sqrt(np.sum((V - ideal_best) ** 2, axis=1))
        d_worst = np.sqrt(np.sum((V - ideal_worst) ** 2, axis=1))

        # Step 5: TOPSIS closeness coefficient
        topsis_score = d_worst / (d_best + d_worst + eps)

        ranked = []
        for i, r in enumerate(valid):
            r = dict(r)
            r["topsis_d_best"] = float(d_best[i])
            r["topsis_d_worst"] = float(d_worst[i])
            r["topsis_score"] = float(topsis_score[i])
            r["topsis_weights"] = {
                "log_proxy_score": 0.4,
                "zico": 0.4,
                "flops_billion": 0.2,
            }
            ranked.append(r)

        ranked = sorted(ranked, key=lambda r: r["topsis_score"], reverse=True)

        return ranked
    def plot_nsga2_fronts(
        self,
        valid,
        selected,
        save_path="nsga2_3d_front_surfaces.png",
        max_fronts_to_plot=4,
        surface_alpha=0.18,
    ):
        import matplotlib
        matplotlib.use("Agg")

        import numpy as np
        import matplotlib.pyplot as plt
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

        if len(valid) == 0:
            print("No valid architectures to plot.")
            return

        # Internal NSGA-II objective matrix.
        # pymoo minimizes all objectives:
        #   -log_proxy_score  -> maximize SynFlow
        #    flops_billion    -> minimize FLOPs
        #   -zico             -> maximize ZiCO
        F = np.array([
            [
                -float(r["log_proxy_score"]),
                float(r["flops_billion"]),
                -float(r["zico"]),
            ]
            for r in valid
        ], dtype=float)

        # Get all nondominated sorting fronts, not just Pareto front 0.
        fronts = NonDominatedSorting().do(
            F,
            only_non_dominated_front=False
        )

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")

        max_fronts_to_plot = min(max_fronts_to_plot, len(fronts))

        for front_rank, front_indices in enumerate(fronts[:max_fronts_to_plot]):
            front_records = [valid[i] for i in front_indices]

            x = np.array([float(r["flops_billion"]) for r in front_records], dtype=float)
            y = np.array([float(r["log_proxy_score"]) for r in front_records], dtype=float)
            z = np.array([float(r["zico"]) for r in front_records], dtype=float)

            # Front 0 should be visually strongest.
            if front_rank == 0:
                label = "Front 0 / Pareto front"
                point_size = 55
                line_width = 0.9
                alpha_points = 1.0
                alpha_surface = surface_alpha + 0.10
            else:
                label = f"Front {front_rank}"
                point_size = 32
                line_width = 0.5
                alpha_points = 0.65
                alpha_surface = surface_alpha

            # Cloud points for this front.
            ax.scatter(
                x,
                y,
                z,
                s=point_size,
                alpha=alpha_points,
                label=label
            )

            # Sheet-like triangulated surface.
            # Needs at least 3 points to form triangles.
            if len(front_records) >= 3:
                try:
                    ax.plot_trisurf(
                        x,
                        y,
                        z,
                        alpha=alpha_surface,
                        linewidth=line_width,
                        edgecolor="black",
                        antialiased=True
                    )
                except Exception as e:
                    print(f"Could not draw trisurf for Front {front_rank}: {e}")

            # Add a visible front curve by sorting along FLOPs.
            # This helps even when the surface is sparse.
            if len(front_records) >= 2:
                order = np.argsort(x)
                ax.plot(
                    x[order],
                    y[order],
                    z[order],
                    linewidth=2.2 if front_rank == 0 else 1.2,
                    alpha=0.95 if front_rank == 0 else 0.55
                )

        # Mark TOPSIS-selected architectures.
        if selected is not None and len(selected) > 0:
            sx = np.array([float(r["flops_billion"]) for r in selected], dtype=float)
            sy = np.array([float(r["log_proxy_score"]) for r in selected], dtype=float)
            sz = np.array([float(r["zico"]) for r in selected], dtype=float)

            ax.scatter(
                sx,
                sy,
                sz,
                marker="*",
                s=300,
                edgecolors="black",
                linewidths=1.2,
                label="TOPSIS selected",
                zorder=20
            )

            for idx, r in enumerate(selected, start=1):
                ax.text(
                    float(r["flops_billion"]),
                    float(r["log_proxy_score"]),
                    float(r["zico"]),
                    f"  TOPSIS #{idx}",
                    fontsize=9
                )

        ax.set_xlabel("FLOPs (B, lower is better)")
        ax.set_ylabel("log10(SynFlow, higher is better)")
        ax.set_zlabel("ZiCO (higher is better)")

        ax.set_title(
            "NSGA-II nondominated fronts with TOPSIS-selected architectures"
        )

        # Adjust view angle for a clearer sheet/cloud look.
        ax.view_init(elev=24, azim=-135)

        ax.grid(True)
        ax.legend(loc="best", fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved 3D NSGA-II front surface plot to: {save_path}")

    def records_to_objective_matrix(self, records):
        """
        Convert evaluated architecture records into the NSGA-II minimization objective matrix.

        Objectives:
            f1 = -log10(SynFlow)  -> minimize because larger SynFlow is better
            f2 = FLOPs            -> minimize because smaller FLOPs is better
            f3 = -ZiCO            -> minimize because larger ZiCO is better
        """
        F = []

        for r in records:
            if (
                "log_proxy_score" in r
                and "flops_billion" in r
                and "zico" in r
                and np.isfinite(float(r["log_proxy_score"]))
                and np.isfinite(float(r["flops_billion"]))
                and np.isfinite(float(r["zico"]))
            ):
                F.append([
                    -float(r["log_proxy_score"]),
                    float(r["flops_billion"]),
                    -float(r["zico"])
                ])

        return np.array(F, dtype=float)


    def normalize_objectives(self, F, ideal=None, nadir=None):
        """
        Normalize objective values into [0, 1].

        Since all objectives are minimization objectives:
            0 is best
            1 is worst
        """
        eps = 1e-12

        if ideal is None:
            ideal = np.min(F, axis=0)

        if nadir is None:
            nadir = np.max(F, axis=0)

        F_norm = (F - ideal) / (nadir - ideal + eps)
        F_norm = np.clip(F_norm, 0.0, 1.0)

        return F_norm, ideal, nadir


    def get_nondominated_records(self, records):
        """
        Extract nondominated records from a list of evaluated architectures.
        """
        valid = []

        for r in records:
            if (
                "log_proxy_score" in r
                and "flops_billion" in r
                and "zico" in r
                and np.isfinite(float(r["log_proxy_score"]))
                and np.isfinite(float(r["flops_billion"]))
                and np.isfinite(float(r["zico"]))
            ):
                valid.append(r)

        if len(valid) == 0:
            return []

        F = self.records_to_objective_matrix(valid)
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)

        return [valid[i] for i in nd_idx]


    def compute_generational_distance(self, front_norm, reference_norm):
        """
        Compute Generational Distance.

        GD measures how close the obtained Pareto front is to a reference Pareto front.

        Lower GD is better.
        If GD = 0, the obtained front exactly matches the reference front.
        """
        if len(front_norm) == 0 or len(reference_norm) == 0:
            return None

        distances = []

        for p in front_norm:
            d = np.linalg.norm(reference_norm - p, axis=1)
            distances.append(np.min(d))

        distances = np.array(distances, dtype=float)

        gd = np.sqrt(np.mean(distances ** 2))
        return float(gd)


    def compute_overall_pareto_spread(self, front_norm):
        """
        Compute Overall Pareto Spread.

        This measures how widely the Pareto front covers the objective space.

        Here OS is computed as the average normalized range across objectives:

            OS = mean(max(f_j) - min(f_j))

        Higher OS means better spread/diversity.
        """
        if len(front_norm) == 0:
            return None

        ranges = np.max(front_norm, axis=0) - np.min(front_norm, axis=0)
        os_value = np.mean(ranges)

        return float(os_value)


    def evaluate_pareto_front_metrics(
        self,
        pareto_records,
        all_records=None,
        reference_records=None,
        save_path="nsga2_pareto_metrics.json"
    ):
        """
        Evaluate Pareto-front quality using:
            1. Hypervolume
            2. Generational Distance
            3. Overall Pareto Spread

        pareto_records:
            The obtained nondominated Pareto front.

        all_records:
            All evaluated architectures. Used to normalize the objectives.

        reference_records:
            Reference Pareto front for GD. If None, the nondominated front from all_records is used.
        """
        import json

        if all_records is None:
            all_records = pareto_records

        if reference_records is None:
            reference_records = self.get_nondominated_records(all_records)

        F_all = self.records_to_objective_matrix(all_records)
        F_pareto = self.records_to_objective_matrix(pareto_records)
        F_ref = self.records_to_objective_matrix(reference_records)

        if len(F_pareto) == 0:
            raise RuntimeError("Cannot compute Pareto metrics because the Pareto front is empty.")

        # Normalize using all evaluated architectures so the scale is consistent.
        _, ideal, nadir = self.normalize_objectives(F_all)
        F_pareto_norm, _, _ = self.normalize_objectives(F_pareto, ideal=ideal, nadir=nadir)
        F_ref_norm, _, _ = self.normalize_objectives(F_ref, ideal=ideal, nadir=nadir)

        # Hypervolume: use a reference point worse than all normalized objective values.
        # Since normalized minimization values are in [0, 1], [1.1, 1.1, 1.1] is safely worse.
        ref_point = np.ones(F_pareto_norm.shape[1]) * 1.1
        hv_indicator = HV(ref_point=ref_point)
        hypervolume = float(hv_indicator(F_pareto_norm))

        # Generational Distance
        gd = self.compute_generational_distance(F_pareto_norm, F_ref_norm)

        # Overall Pareto Spread
        os_value = self.compute_overall_pareto_spread(F_pareto_norm)

        metrics = {
            "hypervolume": hypervolume,
            "generational_distance": gd,
            "overall_pareto_spread": os_value,
            "num_pareto_solutions": int(len(F_pareto_norm)),
            "num_reference_solutions": int(len(F_ref_norm)),
            "num_all_valid_solutions": int(len(F_all)),
            "normalization": {
                "ideal_min_objective_values": ideal.tolist(),
                "nadir_max_objective_values": nadir.tolist()
            },
            "reference_point_for_hypervolume": ref_point.tolist(),
            "objective_order": [
                "-log10(SynFlow)",
                "FLOPs",
                "-ZiCO"
            ],
            "objective_direction": [
                "minimize",
                "minimize",
                "minimize"
            ]
        }

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n================ Pareto Front Metrics ================")
        print(f"Hypervolume              : {hypervolume:.6f}  higher is better")
        print(f"Generational Distance    : {gd:.6f}  lower is better")
        print(f"Overall Pareto Spread OS : {os_value:.6f}  higher is better")
        print(f"Pareto solutions         : {len(F_pareto_norm)}")
        print(f"Saved metrics to         : {save_path}")
        print("======================================================\n")

        return metrics
    def select_nsga2_architectures(self, records, top_k=1):
        # existing valid-record filtering
        valid = []

        for r in records:
            if (
                "log_proxy_score" in r
                and "zico" in r
                and "flops_billion" in r
                and np.isfinite(float(r["log_proxy_score"]))
                and np.isfinite(float(r["zico"]))
                and np.isfinite(float(r["flops_billion"]))
            ):
                valid.append(r)

        if len(valid) == 0:
            raise RuntimeError("No valid NSGA-II records found.")

        # Pareto front extraction
        F = np.array([
            [-r["log_proxy_score"], r["flops_billion"], -r["zico"]]
            for r in valid
        ], dtype=float)

        nd_idx = NonDominatedSorting().do(
            F,
            only_non_dominated_front=True
        )

        pareto = [valid[i] for i in nd_idx]

        # TOPSIS selection goes here
        ranked_pareto = self.rank_architectures_topsis(
            pareto,
            weights=(0.4, 0.4, 0.2),
            criteria=("log_proxy_score", "zico", "flops_billion"),
            benefit=(True, True, False),
        )

        selected_top2 = ranked_pareto[:top_k]

        selected = {
            "selection_method": "TOPSIS",
            "topsis_criteria": ["log_proxy_score", "zico", "flops_billion"],
            "topsis_weights": {
                "log_proxy_score": 0.4,
                "zico": 0.4,
                "flops_billion": 0.2,
            },
            "topsis_criteria_type": {
                "log_proxy_score": "benefit",
                "zico": "benefit",
                "flops_billion": "cost",
            },
            "selected_architectures": selected_top2,
            "ranked_pareto_front": ranked_pareto,
            "pareto_front": pareto,
            "all_valid_architectures": valid,
        }

        with open("nsga2_selected_architecture.json", "w") as f:
            json.dump(selected, f, indent=2, default=str)

        print("\n================ TOPSIS SELECTED ARCHITECTURES ================")
        print("Weights: SynFlow/log_proxy_score = 0.4, ZiCO = 0.4, FLOPs = 0.2")

        for i, r in enumerate(selected_top2, start=1):
            print(f"\nSELECTED ARCHITECTURE #{i}")
            print(f"TOPSIS score: {r['topsis_score']:.6f}")
            print(f"Distance to ideal best: {r['topsis_d_best']:.6f}")
            print(f"Distance to ideal worst: {r['topsis_d_worst']:.6f}")
            print(f"SynFlow: {r['proxy_score']}")
            print(f"log10(SynFlow): {r['log_proxy_score']}")
            print(f"ZiCO: {r['zico']}")
            print(f"FLOPs: {r['flops_billion']:.6f} B")
            print(f"Params: {r['params_million']:.6f} M")
            print(f"MACs: {r['macs_billion']:.6f} B")
            print(f"Latency: {r['latency']:.4f} ms")
            print(f"individual: {r['individual']}")
            print(f"decoded_cell: {r['decoded_cell']}")

        print("\nSaved TOPSIS-selected architecture to: nsga2_selected_architecture.json")

        self.plot_nsga2_fronts(valid=valid, selected=selected_top2, save_path="nsga2_ranked_fronts.png")

        return selected



    def evaluate_architecture_metrics(self, solution, proxy_name="synflow"):
        individual, decoded_cell, decoded_model, n_classes = self.build_model_from_solution(solution)

        measures = self.evaluator.evaluate_zero_cost(
            decoded_model,
            self.epochs,
            n_classes
        )

        proxy_score = float(measures[proxy_name])
        flops = float(measures["flops"])
        zico_score = float(measures["zico"])
        params = float(measures.get("params", 0.0))
        macs = float(measures.get("macs", 0.0))
        sizemb = float(measures.get("sizemb", 0.0))
        latency = float(measures.get("latency", measures.get("Latency", 0.0)))

        safe_proxy = max(proxy_score, 1e-300)

        record = {
            "solution": [float(v) for v in solution],
            "individual": [
                float(v) if isinstance(v, (float, np.floating)) else int(v)
                for v in individual
            ],
            "decoded_cell": repr(decoded_cell),
            "proxy_name": proxy_name,
            "proxy_score": proxy_score,
            "log_proxy_score": float(math.log10(safe_proxy)),
            "flops": flops,
            "flops_billion": flops / 1e9,
            "params": params,
            "params_million": params / 1e6,
            "macs": macs,
            "macs_billion": macs / 1e9,
            "sizemb": sizemb,
            "latency": latency,
            "zico": zico_score,
        }

        self.last_eval_record = record
        return record
    
 
    
    def nsga2_evolve(self, pop_size=10, n_gen=2, seed=1, proxy_name="synflow"): #old evolve algorithm is kept for backwards compatibility not removed yet
        self.nsga2_archive = []

        n_var = 48

        problem = NASNSGA2Problem(
            soga=self,
            n_var=n_var,
            proxy_name=proxy_name
        )

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=TwoPointCrossover(prob=0.9),
            mutation=PolynomialMutation(prob=1.0 / n_var),
            eliminate_duplicates=True
        )

        results = minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", n_gen),
            seed=seed,
            save_history=True,
            verbose=True
        )

        print("\nNSGA-II Pareto objective values:")
        print(results.F)

        selected = self.select_nsga2_architectures(
            self.nsga2_archive,
            top_k=1
        )

        return selected
    
    def __init__(self, population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
                 in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
                 multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption=False,evaluation_type='zero_cost'):
        super().__init__(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size,
                         num_classes, in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain,
                         resume_train, cutout, multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type)

    
    """def evaluate_fitness_single_mealpy(self, solution):
        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])


        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
                individual.append(math.floor(solution[i]*len(self.attentions)))
        individual.append(int(math.floor(2+((self.layers-2)*solution[-1]))))
        print(individual[-1])
        individuals = np.asarray(individual)

        is_final = False
        #print(decode_operations(individual, self.pop.indexes))
        decoded_individual = NetworkCIFAR(self.n_channels, n_classes, individual[-1], True,
                                            decode_cell(decode_operations(individual[:-1], self.pop.indexes)),self.is_medmnist,is_final,
                                            self.dropout_rate, 'FP32', False)
        # decoded_individual = NetworkCIFAR(self.n_channels, n_classes, self.layers, True,
        #                                     decode_cell(decode_operations(individual, self.pop.indexes)),self.is_medmnist,
        #                                     self.dropout_rate, 'FP32', False)

        loss = self.evaluator.evaluate_zero_cost(decoded_individual, self.epochs,n_classes)
        return  loss['synflow']-old function"""
    
    def evaluate_fitness_single_mealpy(self, solution):
        record = self.evaluate_architecture_metrics(solution, proxy_name="synflow")
        return record["proxy_score"]
    

    def evaluate_ensemble_predictions(self,ensemble,medmnist_dataset):

        return None
    def train_final_individual(self,solution,medmnist_dataset):
        data_flag = self.medmnist_dataset
        output_root = './output'
        info = INFO[self.medmnist_dataset]
        num_epochs = 300
        gpu_ids = '0'
        n_classes = len(info['label'])
        batch_size = 1000
        download = True
        run = 'model1'
        individual = []
        for i in range(32):
            if i % 2 == 0:
                individual.append(solution[i])
            else:
                individual.append(int(random.choice(self.pop.params_choices[str(i)])))
                # individual.append(random.choice(self.attentions))
                individual.append(math.floor(solution[i] * len(self.attentions)))
        individual.append(int(math.floor(2 + (3 * solution[-1]))))
        print(individual[-1])
        individuals = np.asarray(individual)
        is_final = False
        decoded_individual = NetworkCIFAR(self.n_channels, n_classes, individual[-1], True,
                                          decode_cell(decode_operations(individual[:-1], self.pop.indexes)),
                                          self.is_medmnist,is_final,
                                          self.dropout_rate, 'FP32', False) #self.is_medmnist
        #First Search for augmentation policy
        best_combination = self.evaluator.auto_search_daapolicy(decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='valid', data_flag=data_flag, output_root=output_root,
                                    num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
        # evaluation = 'test'
        #best_combination = None
        loss = self.evaluator.train(best_combination,decoded_individual, 100, hash_indv=None, grad_clip=5, evaluation='test', data_flag=data_flag, output_root=output_root,
                                    num_epochs=num_epochs, gpu_ids=gpu_ids, batch_size=batch_size,is_final=False, download=download, run=run)
        print("loss", loss)




        print("Final loss is ",loss)
    
    def train_selected_nsga2_architectures_with_da(
        self,
        selected=None,
        selected_json="nsga2_selected_two_architectures.json",
        output_root="./output",
        da_search_epochs=100,
        final_train_epochs=300,
        batch_size=128,
        gpu_ids="",
        download=True,
    ):
        """
        Train and evaluate the two architectures selected from the NSGA-II Pareto front.

        For each selected architecture:
            1. rebuild model from saved individual
            2. search best DA policy on validation set
            3. rebuild fresh model
            4. train with best DA policy
            5. evaluate on test set
        """

        if selected is None:
            with open(selected_json, "r") as f:
                selected = json.load(f)

        selected_architectures = selected["selected_architectures"]

        all_results = []

        for idx, record in enumerate(selected_architectures, start=1):
            run_name = f"nsga2_selected_arch_{idx}"
            data_flag = self.medmnist_dataset

            print("\n" + "=" * 80)
            print(f"Training NSGA-II selected architecture #{idx}")
            print("=" * 80)

            print("Selected architecture search metrics:")
            print(f"SynFlow: {record.get('proxy_score')}")
            print(f"log10(SynFlow): {record.get('log_proxy_score')}")
            print(f"ZiCO: {record.get('zico')}")
            print(f"FLOPs_B: {record.get('flops_billion')}")
            print(f"Params_M: {record.get('params_million')}")
            print(f"Balanced distance: {record.get('balanced_distance')}")
            print(f"Balanced score: {record.get('balanced_score')}")

            individual = record["individual"]

            # ------------------------------------------------------------------
            # 1. Build model for DA policy search
            # ------------------------------------------------------------------
            da_model, decoded_cell, n_classes = self.build_model_from_individual(
                individual,
                is_final=False
            )

            # ------------------------------------------------------------------
            # 2. Search best DA policy using validation performance
            # ------------------------------------------------------------------
            best_da_policy = self.evaluator.auto_search_daapolicy(
                da_model,
                da_search_epochs,
                hash_indv=None,
                grad_clip=5,
                evaluation="valid",
                data_flag=data_flag,
                output_root=output_root,
                num_epochs=da_search_epochs,
                gpu_ids=gpu_ids,
                batch_size=batch_size,
                is_final=False,
                download=download,
                run=run_name + "_da_search"
            )

            print(f"\nBest DA policy for selected architecture #{idx}:")
            print(best_da_policy)

            # Free the DA-search model before creating the final-training model
            del da_model
            torch.cuda.empty_cache()
            gc.collect()

            # ------------------------------------------------------------------
            # 3. Rebuild fresh model for final training, do not reuse da_model because da search may have trained/modified it
            # ------------------------------------------------------------------
            final_model, _, _ = self.build_model_from_individual(
                individual,
                is_final=True
            )

            # ------------------------------------------------------------------
            # 4. Train final model with selected DA policy
            # ------------------------------------------------------------------
            test_result = self.evaluator.train(
                best_da_policy,
                final_model,
                final_train_epochs,
                hash_indv=None,
                grad_clip=5,
                evaluation="test",
                data_flag=data_flag,
                output_root=output_root,
                num_epochs=final_train_epochs,
                gpu_ids=gpu_ids,
                batch_size=batch_size,
                is_final=True, #important so evaluator.train() appllies the chosen da policy
                download=download,
                run=run_name + "_final_test"
            )

            result = {
                "rank": idx,
                "run_name": run_name,
                "search_record": record,
                "best_da_policy": str(best_da_policy),
                "test_result": test_result,
                "decoded_cell": repr(decoded_cell),
                "individual": individual,
            }

            all_results.append(result)

            print(f"\nFinal test result for selected architecture #{idx}:")
            print(test_result)

        with open("nsga2_selected_two_da_training_results.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print("\nSaved DA training results to:")
        print("nsga2_selected_two_da_training_results.json")

        return all_results

    def mealypy_evolve(self, algorithm, pop_size=15, epoch=20,medmnist_dataset=None):

        ## Design a problem dictionary for multiple objective functions above
        problem_multi = {
            "fit_func": self.evaluate_fitness_single_mealpy,
            "lb": [0 for i in range(48)],
            "ub": [0.99 for i in range(48)],
            "minmax": "max",
            "obj_weights": [1],  # Define it or default value will be [1, 1, 1]
            "save_population": True,
            "log_to": "file",
            "log_file": "result.log",  # Default value = "mealpy.log"
        }
        paras_de = {
            "epoch": [100],
            "pop_size": [100],
            "wf": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "cr": [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
            "strategy": [1]
        }
        # term = {
        #     "max_epoch": 2
        # }

        max_time = 60
        term_dict = {
            "max_time": max_time  # 60 seconds to run this algorithm only
        }
        surrogate_params = {'bootstrap': False, 'max_depth': 90, 'max_features': 'sqrt',
                            'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 1800}

        if algorithm == 'pso':
            model = PSO.OriginalPSO(epoch=50, pop_size=50)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        elif algorithm == 'de':
            wf = 0.7
            cr = 0.9
            strategy = 0
            model = BaseDE(epoch, pop_size, wf, cr, strategy)
            #tuner = Tuner(model, paras_de)
            #tuner.execute(problem=problem_multi, n_trials=5, n_jobs=6, mode="parallel", n_workers=6, verbose=True)
            # best_position, best_fitness = model.solve(problem=problem_multi, surrogate_params=surrogate_params,
            #                                           termination=term_dict)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")

            # Store the fitness history of each model
            de_fitness_history = model.history.list_global_best_fit

            # Create convergence chart for each model
            plt.plot(de_fitness_history, label='DE')

            # Add legend and labels
            plt.legend()
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title('Convergence Chart')

            # Display the chart
            plt.show()
            ## self.train_final_individual(best_position,medmnist_dataset)
        elif algorithm == 'lshade':
            #Trying to create an ensemble of 5 top find networks to improve the perforamnce
            ensemble_models = []
            for i in range(2):
                miu_f = 0.5
                miu_cr = 0.5
                model = L_SHADE(epoch, pop_size, miu_f, miu_cr)
                best_position, best_fitness = model.solve(problem=problem_multi)
                print(f"Solution: {best_position}, Fitness: {best_fitness}")
                ensemble_models.append(best_position)
                ## self.train_final_individual(best_position, medmnist_dataset)
            self.evaluate_ensemble_predictions(ensemble_models,medmnist_dataset)
            print(ensemble_models)
            print("Now ensemble predictions from test test")
        elif algorithm == 'ga':
            pc = 0.9
            pm = 0.05
            model1 = BaseGA(epoch, pop_size, pc, pm)
            best_position, best_fitness = model1.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        elif algorithm == 'cmaes':
            model = CMA_ES(epoch, pop_size)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        elif algorithm == 'aco':
            sample_count = 25
            intent_factor = 0.5
            zeta = 1.0
            model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
            best_position, best_fitness = model.solve(problem=problem_multi)
            print(f"Solution: {best_position}, Fitness: {best_fitness}")
        else:
            print("error")
        ## Define the model and solve the problem

        ## Save model to file
        io.save_model(model, "results/model.pkl")
        ## You can access them all via object "history" like this:
        model.history.save_global_objectives_chart(filename="hello/goc")
        model.history.save_local_objectives_chart(filename="hello/loc")

        model.history.save_global_best_fitness_chart(filename="hello/gbfc")
        model.history.save_local_best_fitness_chart(filename="hello/lbfc")

        model.history.save_runtime_chart(filename="hello/rtc")

        model.history.save_exploration_exploitation_chart(filename="hello/eec")

        model.history.save_diversity_chart(filename="hello/dc")

        model.history.save_trajectory_chart(list_agent_idx=[3, 5], selected_dimensions=[3], filename="hello/tc")
    def evolve(self):
        pop_size = 5
        seed = 50
        n_gens = 5
        objectives_list = ['synflow', 'params']
        xl= [ 0.0 for i in range(48)]
        xu= [ 0.99 for i in range(48)]
        xl = np.asarray(xl)
        xu = np.asarray(xu)
        n_obj = len(objectives_list)
        n_var = 48  # NATS-Bench
        problem = NAS(objectives_list=objectives_list, n_var=n_var,
                      n_obj=n_obj,
                      xl=xl, xu=xu,pop = self.pop,population_size=self.population_size, number_of_generations = self.number_of_generations, crossover_prob = self.crossover_prob, mutation_prob= self.mutation_prob, blocks_size=self.blocks_size,
                         num_classes=self.num_classes, in_channels=self.in_channels, epochs=self.epochs, batch_size=self.batch_size, layers=self.layers, n_channels=self.n_channels, dropout_rate=self.dropout_rate, retrain=self.retrain,
                         resume_train=self.resume_train, cutout=self.cutout, multigpu_num=self.multigpu_num)

        algorithm = NSGA2(pop_size=pop_size,
                          sampling=FloatRandomSampling(),
                          crossover=TwoPointCrossover(prob=0.9),
                          mutation=PolynomialMutation(prob=1.0 / n_var),
                          eliminate_duplicates=True)

        stop_criteria = ('n_gen', n_gens)

        results = minimize(
            problem=problem,
            algorithm=algorithm,
            seed=seed,
            save_history=True,
            termination=stop_criteria
        )
        print(results.F)

        # Assuming 'results' contains your optimization results
        # results.F contains the objective values

        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation
        hist_cv = []  # constraint violation in each generation
        hist_cv_avg = []  # average constraint violation in the whole population

        for algo in hist:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

            # retrieve the optimum from the algorithm
            opt = algo.opt

            # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
