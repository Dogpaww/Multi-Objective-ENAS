import argparse
import random
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
import utils
import time
#from ga import GA
#from mo_ga import MOGA
#from mo_gad import MOGADE
from so_ga import SOGA

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--workers', type=int, default=8, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size_train', type=int, default=32, help='batch size')
parser.add_argument('--batch_size_val', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')

parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='EEEA_C', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
parser.add_argument('--mode', type=str, default='FP32', choices=['FP32', 'FP16', 'amp'])
parser.add_argument('--tensorboard', help='Log progress to TensorBoard', default=True, action='store_true')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100','cinic10'])
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--duplicates', type=int, default=1, help='duplicates')

parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'amsbound'])

parser.add_argument('--warmup', action='store_true', default=False, help='use warmup')

parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--increment', type=int, default=6, help='filter increment')
parser.add_argument('--pyramid', action='store_true', default=False, help='pyramid')

parser.add_argument('--autoaugment', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--se', action='store_true', default=False, help='use se')
parser.add_argument('--cutmix', action='store_true', default=False, help='use cutmix')

args = parser.parse_args()

args = utils.get_classes(args)

if __name__ == '__main__':
  #Setting the paramters for the GA-NAS algorithm
  #ga = GA(args.population, args.generations, args.crossover_prob, args.mutate_prob, args.n_blocks, args.classes, 3, 20,
          #args.batch_size_train, args.layers, args.init_channels, args.dropout_rate, True, True, args.cutout, 3)
  population_size =500
  number_of_generations =100
  crossover_prob =0.9
  mutation_prob =0.6
  blocks_size =32
  num_classes =4 # 3 for covid,7 for ham10000, 4 for ocular_toxoplosmosis
  in_channels =3
  epochs =4
  batch_size = 1024
  layers = 7
  n_channels = 16
  dropout_rate = 0.3
  retrain = True
  resume_train = False
  cutout = False
  multigpu_num = 1
  medmnist_dataset = 'breastmnist'
  check_power_consumption = False
  is_medmnist = True
  evaluation_type = 'zero_cost' #Zero Cost or Training
  is_tune= False
  soga_algorithm = 'lshade' #de,ga,ea,pso,aco,sa,ba,gwo,ba
  #Multi-Objective
  # ga = MOGA(population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,retrain,resume_train,cutout,multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type)
  # # #Running the algorithm
  # ga.evolve()
  #Single Objective
  #Implementation of MOEA/D algorithm
  #ga = MOGADE(population_size, number_of_generations, crossover_prob, mutation_prob, blocks_size, num_classes,
            # in_channels, epochs, batch_size, layers, n_channels, dropout_rate, retrain, resume_train, cutout,
            # multigpu_num, medmnist_dataset, is_medmnist, check_power_consumption,evaluation_type)
  # #Running the algorithm
  #ga.evolve()
  ga = SOGA(population_size,number_of_generations,crossover_prob,mutation_prob,blocks_size,num_classes,in_channels,epochs,batch_size,layers,n_channels,dropout_rate,retrain,resume_train,cutout,multigpu_num,medmnist_dataset,is_medmnist,check_power_consumption,evaluation_type)
  #Running the algorithm
  #ga.mealypy_evolve('de',10,2,medmnist_dataset)



  # Record the start time
  start_time = time.time()

  RUN_NSGA2_SEARCH = True

  
  SMOKE_TEST=False 

  # Running the algorithm
  #ga.mealypy_evolve('de',10,2,medmnist_dataset) -> old call
  if RUN_NSGA2_SEARCH:
    selected = ga.nsga2_evolve(
      pop_size=20,
      n_gen=10,
      seed=1,
      proxy_name="synflow")

   #for smoke test pop size=10 n_gen=2,real run =20,10
  # Record the end time
  RUN_DA_TRAINING = True
  if RUN_DA_TRAINING:
    ga.train_selected_nsga2_architectures_with_da(
        selected=selected,
        da_search_epochs=1 if SMOKE_TEST else 100,
        final_train_epochs=1 if SMOKE_TEST else 300,
        batch_size=32 if SMOKE_TEST else 64,
        gpu_ids="-1" if SMOKE_TEST else "0",
    )
  
  end_time = time.time()

  # Calculate the elapsed time
  elapsed_time = end_time - start_time
  print(f"Elapsed time: {elapsed_time} seconds")