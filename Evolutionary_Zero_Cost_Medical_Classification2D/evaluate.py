import logging
import os.path

import torchattacks
from acsconv.converters import ACSConverter, Conv2_5dConverter, Conv3dConverter
from art.estimators.classification import PyTorchClassifier
from medmnist import INFO
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import gc
from dataset import Dataset
import random
import numpy as np
import random
import torch
from medmnist.info import INFO, DEFAULT_ROOT
from fvcore.nn import FlopCountAnalysis
import torch.autograd.profiler as profiler
import torchvision
from collections import OrderedDict
import torchvision.transforms as transforms
import torch.nn as nn
from evaluation_measures import evaluate_measures
import torch.nn.functional as F
import torch.optim as optim
from thop import profile
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary
from foresight.models import *
from foresight.pruners import *
from foresight.dataset import *
from foresight.pruners import predictive
from foresight.weight_initializers import init_net
import utils
import json
import time
import medmnist
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torchvision import datasets
from tqdm import trange

import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0

# Define your data augmentation policies
"""data_augmentations = [
    transforms.RandomApply([transforms.RandomHorizontalFlip()]),
    transforms.RandomApply([transforms.RandomVerticalFlip()]),
    transforms.RandomApply([transforms.RandomRotation(30)]),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)]),
    transforms.RandomApply([transforms.RandomResizedCrop(32)]),
    transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))]),
    transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))])
]"""
def get_data_augmentations(input_size):
    return [
        transforms.RandomApply([transforms.RandomHorizontalFlip()]),
        transforms.RandomApply([transforms.RandomVerticalFlip()]),
        transforms.RandomApply([transforms.RandomRotation(30)]),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2
            )
        ]),
        transforms.Compose([
    transforms.RandomApply([
        transforms.RandomResizedCrop((input_size, input_size))
    ]),
    transforms.Resize((input_size, input_size))
]),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))
        ]),
        transforms.RandomApply([
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3)
            )
        ])
    ]

# Define the number of augmentations to select
num_to_select = 2

# Define the number of generations and population size
num_generations = 5
population_size = 5


class Evaluate:
    def __init__(self, batch_size,medmnist_dataset,is_medmnist,check_power_consumption):
        self.dataset = Dataset()
        self.batch_size = batch_size
        self.medmnist_dataset = medmnist_dataset
        self.criterion = nn.CrossEntropyLoss()
        self.check_power_consumption = check_power_consumption
        self.is_medmnist = is_medmnist
        self.optimizer = None
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.inbreast_dataset(self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.breast_dataset_mias(
             #self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.arabic(
            #self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.breast_dataset_ddsm(
             #self.batch_size)
        # self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.combined_breast_datasets(
        #      self.batch_size)
        #self.train_loader, self.valid_loader, self.test_loader, self.classes = self.dataset.kvasir_dataset(self.batch_size)
        #self.train_loader,  self.test_loader, self.classes = self.dataset.covid_radiographic_dataset(self.batch_size)
        #self.train_loader, self.test_loader = self.dataset.ham10000(self.batch_size)
        #self.train_loader, self.test_loader = self.dataset.ocular_toxoplosmosis(self.batch_size)
        if self.is_medmnist == True:
            self.train_loader, self.valid_loader, self.test_loader = self.dataset.get_dataset_medmnist(self.medmnist_dataset,
                                                                                                       self.batch_size)
        self.lr = 0.001
        self.scheduler = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate_architecture(self,candidate, data_augmentations, model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run):
        # Apply data augmentations to your data
        composed_transform = transforms.Compose(data_augmentations)
        print(composed_transform)
        # Assume this function trains a model and returns a fitness score
        # You need to replace 'your_data' with your actual data
        # your_data_transformed = composed_transform(your_data)
        fitn = self.train(composed_transform,model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run)
        return fitn,data_augmentations

    # Define the fitness function
    def fitness_function(self,candidate,model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run):
        # Assume evaluate_architecture returns a fitness score
        return self.evaluate_architecture(candidate,candidate, model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run)

    # Genetic Algorithm
    def genetic_algorithm(self,model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run,input_size=32, accumulation_steps=1, use_amp=True):
        # Generate an initial population
        #population = [random.sample(data_augmentations, num_to_select) for _ in range(population_size)]

        aug_pool = get_data_augmentations(input_size)
        population = [random.sample(aug_pool, num_to_select) for _ in range(population_size)]
        base_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if not (k.endswith("total_ops") or k.endswith("total_params"))
        }
        fitness_arc, data_aug_arc = [],[]
        for generation in range(num_generations):
            # Evaluate the fitness of each individual in the population
            fitness_aug = []

            for candidate in population:
                candidate_model = copy.deepcopy(model).cpu()
                candidate_model.load_state_dict(base_state, strict=False)
                try:
                    result = self.fitness_function(
                        candidate,
                        model,
                        epochs,
                        hash_indv,
                        grad_clip,
                        evaluation,
                        data_flag,
                        output_root,
                        num_epochs,
                        gpu_ids,
                        batch_size,
                        is_final,
                        download,
                        run
                    )
                except Exception as e:
                    print("DA candidate failed:",e)
                    result=(float("-inf"),candidate)

                fitness_aug.append(result)

                # Clean temporary GPU memory after this DA policy trial
                del result
                del candidate_model
                torch.cuda.empty_cache()
                gc.collect()

                if torch.cuda.is_available():
                    print(
                        "[GPU after DA candidate] "
                        f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f} GB, "
                        f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f} GB"
                    )
            # Separate into two lists
            fitness_scores = [item[0] for item in fitness_aug]
            data_augmentations_archive = [item[1] for item in fitness_aug]
            # Select the top individuals based on fitness scores
            selected_indices = sorted(range(len(fitness_scores)), key=lambda k: fitness_scores[k], reverse=True)[
                               :population_size // 2]
            selected_population = [population[i] for i in selected_indices]

            # Create new individuals through crossover
            new_population = []
            for _ in range(population_size // 2):
                parent1, parent2 = random.sample(selected_population, 2)
                crossover_point = random.randint(0, num_to_select - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
                new_population.append(child)

            # Mutate some individuals
            mutated_population = []
            for i in range(population_size // 5):
                index_to_mutate = random.randint(0, len(selected_population) - 1)
                mutation_point = random.randint(0, num_to_select - 1)

                # Avoid choosing the same policy that already exists in the individual
                available_policies = list(set(aug_pool) - set(selected_population[index_to_mutate]))
                mutated_individual = selected_population[index_to_mutate].copy()
                mutated_individual[mutation_point] = random.choice(available_policies)
                mutated_population.append(mutated_individual)
            fitness_arc.append(fitness_scores)
            data_aug_arc.append(data_augmentations_archive)
            # Combine the selected and newly generated individuals
            population = selected_population + new_population + mutated_population

        fitness_arc = [item for sublist in fitness_arc for item in sublist]
        data_augmentations_archive = [item for sublist in data_aug_arc for item in sublist]

        # Select the best individual as the result
        max_index = fitness_arc.index(max(fitness_arc))

        best_individual = data_augmentations_archive[max_index]
        return best_individual

    def auto_search_daapolicy(self, model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run,input_size=32, accumulation_steps=1, use_amp=True):       # Example usage
        best_combination = self.genetic_algorithm(model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
                batch_size,is_final, download, run,input_size=input_size,accumulation_steps=accumulation_steps,use_amp=use_amp)
        print("Best Data Augmentation Combination:", best_combination)
        return best_combination

    def evaluate_zero_cost(self,model,epochs,n_classes,warmup=False,input_size=32,zc_batch_size=32):
        num_epochs = 10
        as_rgb = True
        resize = False
        lr = 0.001
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[self.medmnist_dataset]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])
        shape_transform = False
        DataClass = getattr(medmnist, info['python_class'])


        #print('==> Preparing data...')
        if '3d' in self.medmnist_dataset:
            train_transform = utils.Transform3D(mul='random') if shape_transform else utils.Transform3D()
            eval_transform = utils.Transform3D(mul='0.5') if shape_transform else utils.Transform3D()

            class ReshapeData(object):
                def __call__(self, sample):
                    # Assuming 'sample' is a 3x28x28x28 tensor
                    sample = sample.view(3, 28, 28)
                    return sample

            download =True
            batch_size = 32
            train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb)
            train_dataset_at_eval = DataClass(split='train', transform=eval_transform, download=download, as_rgb=as_rgb)
            val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb)
            test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb)
        else:
            # Define data augmentations
            data_transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                # transforms.RandomRotation(15),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)), #changed size=32
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # Modify normalization based on your dataset
            ])

            # data_transform = transforms.Compose(
            #     [transforms.ToTensor(),
            #      transforms.Normalize(mean=[.5], std=[.5])])
            download =True
            batch_size = zc_batch_size #changed batch size from 32
            train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
            val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
            test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()


        # Define the loss function and optimizer
        #criterion = nn.CrossEntropyLoss()
        dataload = 'random'
        init_w_type = 'none'
        init_b_type = 'none'
        dataload_info = 4 #use =4 for a more serious run
        # if '3d' in self.medmnist_dataset:
        #     model = ACSConverter(model)

        init_net(model, init_w_type, init_b_type)
        if self.is_medmnist == True:
            measures = predictive.find_measures(model,self.medmnist_dataset,
                                            val_loader,
                                            (dataload, 1, n_classes),
                                            self.device,
                                            loss_fn = criterion,
                                            measure_names={'synflow','zico'}
                                            )
        else:
            measures = predictive.find_measures(model,self.medmnist_dataset,
                                            self.train_loader,
                                            (dataload, 1, n_classes),
                                            self.device,
                                            loss_fn = criterion,
                                            measure_names={'synflow','zico'}
                                            )
        #print(measures)

        #Computing FLOPS and Parameters of model
        if self.is_medmnist == True:
            input_tensor = torch.randn(4, 3, input_size, input_size)  # Replace with your input size
        else:
            input_tensor = torch.randn(4, 3, input_size, input_size)  # Replace with your input size


        #input_tensor = input_tensor.cuda() - works only for cuda
        input_tensor = input_tensor.to(device)
        #model = model.cuda()
        model = model.to(device)
        if self.check_power_consumption == True:
            # Run a few warm-up iterations
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_tensor)

            # Measure the time and profile the inference
            #with profiler.profile(use_cuda=True) as prof: -old code only works with cuda
            with profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                for _ in range(100):  # Adjust the number of iterations as needed
                    with torch.no_grad():
                        output = model(input_tensor)

            # Calculate the total inference time
            total_time_seconds = prof.self_cpu_time_total / 1000.0  # in seconds

            # Assuming you have power consumption information (in watts) for your device
            power_consumption_watts = 5.0  # Replace with the actual power consumption of your device

            # Estimate energy consumption (in joules)
            energy_consumption_joules = total_time_seconds * power_consumption_watts

            # Convert energy consumption to megajoules
            energy_consumption_megajoules = energy_consumption_joules / 1e6

            print(f"Total inference time: {total_time_seconds} seconds")
            print(f"Estimated energy consumption: {energy_consumption_megajoules} megajoules")
        flops, params = profile(model, inputs=(input_tensor,))
        size_in_mb = utils.count_parameters_in_MB(model)
        print(f"FLOPs: {flops / 1e9} billion")
        print(f"Parameters: {params / 1e6} million")
        with torch.no_grad():
            start_time = time.time()
            model(input_tensor)
            end_time = time.time()

        latency = (end_time - start_time) * 1000  # Convert to milliseconds
        print(f"Latency: {latency:.2f} ms")
        macs = flops / 2  # Divide by 2 since one MAC involves one multiplication and one addition
        print(f"MACs: {macs / 1e9} billion")
        # measures['Latency'] = latency
        measures['sizemb'] = size_in_mb
        measures['macs'] = macs
        measures['flops'] = flops
        measures['params'] = params
        measures['latency']=latency #added line

        measures['proxy_name'] = 'synflow' #added line
        measures['proxy_score'] = float(measures['synflow']) #added line
        measures['zico']=float(measures['zico'])


        return measures

            # Training
    def __train(
        self,
        model,
        train_loader,
        task,
        criterion,
        optimizer,
        device,
        writer,
        scaler=None,
        accumulation_steps=1,
        use_amp=True
    ):
        total_loss = []
        global iteration
        grad_clip = 5

        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device, non_blocking=True)
            else:
                targets = torch.squeeze(targets, 1).long().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and torch.cuda.is_available() and device.type== "cuda")):
                outputs, logits_aux = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            total_loss.append(loss.item() * accumulation_steps)
            writer.add_scalar('train_loss_logs', loss.item() * accumulation_steps, iteration)
            iteration += 1

            del inputs, targets, outputs, logits_aux, loss

        # Handle leftover batches if len(train_loader) is not divisible by accumulation_steps
        if len(train_loader) % accumulation_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        epoch_loss = sum(total_loss) / len(total_loss)
        return epoch_loss
    
    def apply_tta(self,image):
        tta_transform = transforms.Compose([
            #Add other transformations as needed
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            #transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),
            # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.2),
            #transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5], std=[0.5])  # Modify normalization based on your dataset
        ])
        augmented_image = tta_transform(image)
        return augmented_image
    def __test(self, model, evaluator, data_loader, task, criterion, device, run, type_task, save_folder=None):
        check_evaluator = medmnist.Evaluator(self.medmnist_dataset, type_task)

        info = INFO[self.medmnist_dataset]
        task = info["task"]

        root = DEFAULT_ROOT
        npz_file = np.load(os.path.join(root, "{}.npz".format(self.medmnist_dataset)))

        if type_task == 'train':
            self.labels = npz_file['train_labels']
        elif type_task == 'val':
            self.labels = npz_file['val_labels']
        elif type_task == 'test':
            self.labels = npz_file['test_labels']
        else:
            raise ValueError

        model.eval()

        total_loss = []
        y_score = []

        with torch.inference_mode():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(device, non_blocking=True)
                outputs, logits_aux = model(inputs)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32).to(device, non_blocking=True)
                    loss = criterion(outputs, targets)
                    outputs = torch.sigmoid(outputs)
                else:
                    targets = torch.squeeze(targets, 1).long().to(device, non_blocking=True)
                    loss = criterion(outputs, targets)
                    outputs = torch.softmax(outputs, dim=1)

                total_loss.append(loss.item())

                # Store predictions on CPU, not GPU
                y_score.append(outputs.detach().cpu())

                del inputs, targets, outputs, logits_aux, loss

        y_score = torch.cat(y_score, dim=0).numpy()

        auc, acc = evaluator.evaluate(y_score, save_folder, run)
        f1 = evaluate_measures(self.labels, y_score, task)
        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc, f1]
    
    def train_ensemble(self, augmented_topology, models, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root,
              num_epochs, gpu_ids,
              batch_size, is_final, download, run):
        # Setting the parameters
        as_rgb = True
        resize = False
        # lr = 0.001 # This is from the medmnist authors
        lr = 0.005  # This is from best reported score paper
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[data_flag]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        print('==> Preparing data...')
        if is_final == True:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(size=(224, 224), antialias=False),
                 transforms.Normalize(mean=[.5], std=[.5])])
        else:
            data_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(size=(32, 32), antialias=False),
                 transforms.Normalize(mean=[.5], std=[.5])])
        if evaluation == 'valid':
            data_transform = transforms.Compose([
                data_transform,
                augmented_topology.transforms[0],
                augmented_topology.transforms[1]
            ])
        else:
            if is_final == False:
                pass
            else:
                data_transform = transforms.Compose([
                    data_transform,
                    augmented_topology[0],
                    augmented_topology[1]
                ])
        train_dataset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)
        val_dataset = DataClass(split='val', transform=data_transform, download=download, as_rgb=as_rgb)
        test_dataset = DataClass(split='test', transform=data_transform, download=download, as_rgb=as_rgb)

        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        print('==> Building and training model...')
        predictions = []
        for model in models:
            model = model.to(device)

            train_evaluator = medmnist.Evaluator(data_flag, 'train')
            val_evaluator = medmnist.Evaluator(data_flag, 'val')
            test_evaluator = medmnist.Evaluator(data_flag, 'test')

            if task == "multi-label, binary-class":
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, 'train',
                                        output_root)
            val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val', output_root)
            test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                       output_root)

            print('train  auc: %.5f  acc: %.5f\n  f1: %.5f\n' % (train_metrics[1], train_metrics[2], train_metrics[3]) + \
                  'val  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (val_metrics[1], val_metrics[2], val_metrics[3]) + \
                  'test  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (test_metrics[1], test_metrics[2], test_metrics[3]))

            if num_epochs == 0:
                return

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=3e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

            logs = ['loss', 'auc', 'acc']
            train_logs = ['train_' + log for log in logs]
            val_logs = ['val_' + log for log in logs]
            test_logs = ['test_' + log for log in logs]
            log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

            writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

            best_auc = 0
            best_epoch = 0
            best_model = model

            global iteration
            iteration = 0
            # Training the models till the given epochs
            for epoch in trange(num_epochs):
                train_loss = self.__train(
                    model,
                    train_loader,
                    task,
                    criterion,
                    optimizer,
                    device,
                    writer,
                    scaler=scaler,
                    accumulation_steps=accumulation_steps,
                    use_amp=use_amp
                )

                train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                            'train')
                val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val')
                test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test')

                scheduler.step()

                for i, key in enumerate(train_logs):
                    log_dict[key] = train_metrics[i]
                for i, key in enumerate(val_logs):
                    log_dict[key] = val_metrics[i]
                for i, key in enumerate(test_logs):
                    log_dict[key] = test_metrics[i]

                for key, value in log_dict.items():
                    writer.add_scalar(key, value, epoch)

                cur_auc = val_metrics[1]
                if cur_auc > best_auc:
                    best_epoch = epoch
                    best_auc = cur_auc
                    best_model = model
                    print('cur_best_auc:', best_auc)
                    print('cur_best_epoch', best_epoch)

            state = {
                'net': best_model.state_dict(),
            }

            path = os.path.join(output_root, 'best_model.pth')
            torch.save(state, path)

        train_metrics = self.__test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                    'train',
                                    output_root)
        val_metrics = self.__test(best_model, val_evaluator, val_loader, task, criterion, device, run, 'val',
                                  output_root)
        test_metrics = self.__test(best_model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (
            train_metrics[1], train_metrics[2], train_metrics[3])
        val_log = 'val  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (val_metrics[1], val_metrics[2], val_metrics[3]) #old code showed the train metric for the last one instead of val
        #test_log = 'test  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (test_metrics[1], test_metrics[2], train_metrics[3])-old code output the train metric instead of test
        test_log = 'test  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (test_metrics[1], test_metrics[2], test_metrics[3])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        print(log)

        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)

        # Computing FLOPS and Parameters of model
        if self.is_medmnist == True:
            input_tensor = torch.randn(4, 3, 32, 32)  # Replace with your input size
        else:
            input_tensor = torch.randn(4, 3, 256, 256)  # Replace with your input size

        input_tensor = input_tensor.cuda()
        model = model.cuda()
        flops, params = profile(model, inputs=(input_tensor,))
        writer.close()

        # Retuning the fitness values
        if evaluation == 'val':
            return val_metrics[1] + test_metrics[2], flops
        else:
            return test_metrics[1] + test_metrics[2]
    def train(self, augmented_topology,model, epochs, hash_indv, grad_clip, evaluation, data_flag, output_root, num_epochs, gpu_ids,
          batch_size,is_final, download, run,input_size=None,accumulation_steps=1,use_amp=True):

        if input_size is None:
            input_size = 224 if is_final else 32
            print(
                f"[TRAIN CONFIG] run={run}, "
                f"is_final={is_final}, "
                f"input_size={input_size}, "
                f"batch_size={batch_size}, "
                f"accumulation_steps={accumulation_steps}, "
                f"use_amp={use_amp}"
            )

        # Setting the parameters
        as_rgb = True
        resize = False
        #lr = 0.001 # This is from the medmnist authors
        lr = 0.025 # This is from best reported score paper
        gamma = 0.1
        milestones = [0.5 * num_epochs, 0.75 * num_epochs]

        info = INFO[data_flag]
        task = info['task']
        n_channels = 3
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])

        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        output_root = os.path.join(output_root, data_flag, time.strftime("%y%m%d_%H%M%S"))
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        print('==> Preparing data...')

        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(input_size, input_size), antialias=False),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

        if evaluation == 'valid':
            train_transform = transforms.Compose([
                base_transform,
                augmented_topology.transforms[0],
                augmented_topology.transforms[1]
            ])
        elif is_final:
            train_transform = transforms.Compose([
                base_transform,
                augmented_topology[0],
                augmented_topology[1]
            ])
        else:
            train_transform = base_transform

        eval_transform = base_transform        
        train_dataset = DataClass(split='train', transform=train_transform, download=download, as_rgb=as_rgb)
        val_dataset = DataClass(split='val', transform=eval_transform, download=download, as_rgb=as_rgb)
        test_dataset = DataClass(split='test', transform=eval_transform, download=download, as_rgb=as_rgb)


        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)

        print('==> Building and training model...')

        model = model.to(device)

        train_evaluator = medmnist.Evaluator(data_flag, 'train')
        val_evaluator = medmnist.Evaluator(data_flag, 'val')
        test_evaluator = medmnist.Evaluator(data_flag, 'test')

        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run, 'train',
                                    output_root)
        val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val', output_root)
        test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        print('train  auc: %.5f  acc: %.5f\n  f1: %.5f\n' % (train_metrics[1], train_metrics[2], train_metrics[3]) + \
              'val  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (val_metrics[1], val_metrics[2], val_metrics[3]) + \
              'test  auc: %.5f  acc: %.5f\n f1: %.5f\n' % (test_metrics[1], test_metrics[2], test_metrics[3]))

        if num_epochs == 0:
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,milestones=milestones,gamma=gamma)

        logs = ['loss', 'auc', 'acc']
        train_logs = ['train_' + log for log in logs]
        val_logs = ['val_' + log for log in logs]
        test_logs = ['test_' + log for log in logs]
        log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

        writer = SummaryWriter(log_dir=os.path.join(output_root, 'Tensorboard_Results'))

        best_auc = 0
        best_epoch = 0
        best_model = model

        #global iteration
        #iteration = 0

        global iteration
        iteration = 0

        use_cuda_amp = (
            use_amp
            and torch.cuda.is_available()
            and device.type == "cuda"
        )

        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda_amp)
        # Training the models till the given epochs
        for epoch in trange(num_epochs):
            train_loss = self.__train(model, train_loader, task, criterion, optimizer, device, writer,scaler=scaler,accumulation_steps=accumulation_steps,use_amp=use_cuda_amp)

            train_metrics = self.__test(model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                        'train')
            val_metrics = self.__test(model, val_evaluator, val_loader, task, criterion, device, run, 'val')
            test_metrics = self.__test(model, test_evaluator, test_loader, task, criterion, device, run, 'test')

            scheduler.step()

            for i, key in enumerate(train_logs):
                log_dict[key] = train_metrics[i]
            for i, key in enumerate(val_logs):
                log_dict[key] = val_metrics[i]
            for i, key in enumerate(test_logs):
                log_dict[key] = test_metrics[i]

            for key, value in log_dict.items():
                writer.add_scalar(key, value, epoch)

            cur_auc = val_metrics[1]
            if cur_auc > best_auc:
                best_epoch = epoch
                best_auc = cur_auc
                best_model = model
                print('cur_best_auc:', best_auc)
                print('cur_best_epoch', best_epoch)

        state = {
            'net': best_model.state_dict(),
        }

        path = os.path.join(output_root, 'best_model.pth')
        torch.save(state, path)

        train_metrics = self.__test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device, run,
                                    'train',
                                    output_root)
        val_metrics = self.__test(best_model, val_evaluator, val_loader, task, criterion, device, run, 'val',
                                  output_root)
        test_metrics = self.__test(best_model, test_evaluator, test_loader, task, criterion, device, run, 'test',
                                   output_root)

        train_log = 'train  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (
            train_metrics[1], train_metrics[2], train_metrics[3])
        val_log = 'val  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (val_metrics[1], val_metrics[2], train_metrics[3])
        test_log = 'test  auc: %.5f  acc: %.5f\n   f1: %.5f\n' % (test_metrics[1], test_metrics[2], train_metrics[3])

        log = '%s\n' % (data_flag) + train_log + val_log + test_log
        print(log)

        with open(os.path.join(output_root, '%s_log.txt' % (data_flag)), 'a') as f:
            f.write(log)

        # Computing FLOPS and Parameters of model
        if self.is_medmnist == True:
                input_tensor = torch.randn(4, 3, 32, 32)  # Replace with your input size
        else:
                input_tensor = torch.randn(4, 3, 256, 256)  # Replace with your input size

        input_tensor=input_tensor.to(device)
        model = model.to(device)
        print(input_tensor.device)
        #print(model.device)
        flops, params = profile(model, inputs=(input_tensor,))
        writer.close()

        # Retuning the fitness values
        if evaluation == 'val':
            return val_metrics[2],flops
        else:
            return test_metrics[1]+test_metrics[2]
