import random

import numpy as np
import torch
import json
import random
import os
from copy import deepcopy

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
    
    def select_nsga2_architectures(self, records, top_k=2):
        valid = []
        seen = set()

        for r in records:
            if r is None:
                continue
            if r.get("error", None) is not None:
                continue

            required_keys = [
                "proxy_score",
                "log_proxy_score",
                "zico",
                "flops",
                "flops_billion",
                "decoded_cell"
            ]

            if any(k not in r for k in required_keys):
                continue

            if not np.isfinite(r["proxy_score"]):
                continue
            if not np.isfinite(r["log_proxy_score"]):
                continue
            if not np.isfinite(r["zico"]):
                continue
            if not np.isfinite(r["flops"]):
                continue

            # avoid duplicate architectures
            key = r["decoded_cell"]
            if key in seen:
                continue

            seen.add(key)
            valid.append(r)

        if len(valid) == 0:
            raise RuntimeError("No valid architectures were evaluated by NSGA-II.")

        # pymoo minimizes all objectives:
        #   -log_proxy_score -> maximize SynFlow
        #   flops_billion    -> minimize FLOPs
        #   -zico            -> maximize ZiCO
        F = np.array([
            [-r["log_proxy_score"], r["flops_billion"], -r["zico"]]
            for r in valid
        ], dtype=float)

        nd_idx = NonDominatedSorting().do(
            F,
            only_non_dominated_front=True
        )

        pareto = [valid[i] for i in nd_idx]

        # Normalize objectives into benefit scores:
        #   proxy_norm = 1 means best SynFlow
        #   zico_norm  = 1 means best ZiCO
        #   flops_norm = 1 means lowest FLOPs
        log_proxy_values = np.array([r["log_proxy_score"] for r in pareto], dtype=float)
        zico_values = np.array([r["zico"] for r in pareto], dtype=float)
        flops_values = np.array([r["flops"] for r in pareto], dtype=float)

        eps = 1e-12

        proxy_min, proxy_max = log_proxy_values.min(), log_proxy_values.max()
        zico_min, zico_max = zico_values.min(), zico_values.max()
        flops_min, flops_max = flops_values.min(), flops_values.max()

        for r in pareto:
            if proxy_max - proxy_min < eps:
                proxy_norm = 1.0
            else:
                proxy_norm = (r["log_proxy_score"] - proxy_min) / (proxy_max - proxy_min + eps)

            if zico_max - zico_min < eps:
                zico_norm = 1.0
            else:
                zico_norm = (r["zico"] - zico_min) / (zico_max - zico_min + eps)

            if flops_max - flops_min < eps:
                flops_norm = 1.0
            else:
                flops_norm = (flops_max - r["flops"]) / (flops_max - flops_min + eps)

            r["proxy_norm"] = float(proxy_norm)
            r["zico_norm"] = float(zico_norm)
            r["flops_norm"] = float(flops_norm)

            r["balanced_distance"] = float(
                math.sqrt(
                    (1.0 - proxy_norm) ** 2 +
                    (1.0 - zico_norm) ** 2 +
                    (1.0 - flops_norm) ** 2
                )
            )

            r["balanced_score"] = float(
                (proxy_norm + zico_norm + flops_norm) / 3.0
            )

        # Select the two Pareto architectures closest to the ideal point.
        selected_two = sorted(
            pareto,
            key=lambda r: (r["balanced_distance"], -r["balanced_score"])
        )[:top_k]

        selected = {
            "selected_architectures": selected_two,
            "pareto_front": pareto,
            "all_valid_architectures": valid,
        }

        with open("nsga2_selected_two_architectures.json", "w") as f:
            json.dump(selected, f, indent=2)

        self.plot_pareto_front(
            valid=valid,
            pareto=pareto,
            selected=selected_two,
            save_path="nsga2_pareto_front.png"
        )

        print("\nSelected two architectures from Pareto front:")

        for i, r in enumerate(selected_two, start=1):
            print(f"\nSELECTED ARCHITECTURE #{i}")
            print(f"proxy_name: {r['proxy_name']}")
            print(f"SynFlow: {r['proxy_score']}")
            print(f"log10(SynFlow): {r['log_proxy_score']}")
            print(f"ZiCO: {r['zico']}")
            print(f"FLOPs: {r['flops_billion']:.6f} B")
            print(f"Params: {r['params_million']:.6f} M")
            print(f"MACs: {r['macs_billion']:.6f} B")
            print(f"Latency: {r['latency']:.4f} ms")
            print(f"proxy_norm: {r['proxy_norm']:.4f}")
            print(f"zico_norm: {r['zico_norm']:.4f}")
            print(f"flops_norm: {r['flops_norm']:.4f}")
            print(f"balanced_distance: {r['balanced_distance']:.4f}")
            print(f"balanced_score: {r['balanced_score']:.4f}")
            print(f"individual: {r['individual']}")
            print(f"decoded_cell: {r['decoded_cell']}")

        print("\nSaved selected architectures to: nsga2_selected_two_architectures.json")
        print("Saved Pareto plot to: nsga2_pareto_front.png")

        return selected
    
    def plot_pareto_front(self, valid, pareto, selected, save_path="nsga2_pareto_front.png"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_flops = [r["flops_billion"] for r in valid]
        all_proxy = [r["log_proxy_score"] for r in valid]
        all_zico = [r["zico"] for r in valid]

        pareto_flops = [r["flops_billion"] for r in pareto]
        pareto_proxy = [r["log_proxy_score"] for r in pareto]
        pareto_zico = [r["zico"] for r in pareto]

        selected_flops = [r["flops_billion"] for r in selected]
        selected_proxy = [r["log_proxy_score"] for r in selected]
        selected_zico = [r["zico"] for r in selected]

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            all_flops,
            all_proxy,
            all_zico,
            alpha=0.25,
            label="All valid architectures"
        )

        ax.scatter(
            pareto_flops,
            pareto_proxy,
            pareto_zico,
            s=45,
            label="Pareto front"
        )

        ax.scatter(
            selected_flops,
            selected_proxy,
            selected_zico,
            s=140,
            marker="*",
            label="Selected top 2"
        )

        ax.set_xlabel("FLOPs (billions, lower is better)")
        ax.set_ylabel("log10(SynFlow, higher is better)")
        ax.set_zlabel("ZiCO (higher is better)")
        ax.set_title("NSGA-II Pareto front: SynFlow vs FLOPs vs ZiCO")

        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    
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
            top_k=2
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
