
import numpy as np
import torch
import torch.nn as nn
from medmnist import INFO

from . import measure


def _get_logits(outputs):
    """
    NetworkCIFAR often returns:
        outputs, features

    Official ZiCO assumes:
        outputs only

    This helper makes ZiCO compatible with model.
    """
    if isinstance(outputs, tuple):
        return outputs[0]
    return outputs


def _prepare_targets(targets, medmnist_dataset, device):
    """
    Match MedMNIST target formatting.

    Multi-class:
        CrossEntropyLoss expects LongTensor of shape [batch]

    Multi-label binary:
        BCEWithLogitsLoss expects FloatTensor
    """
    info = INFO[medmnist_dataset]
    task = info["task"]

    if task == "multi-label, binary-class":
        return targets.to(torch.float32).to(device)

    if targets.ndim > 1 and targets.shape[1] == 1:
        targets = targets.squeeze(1)

    return targets.long().to(device)


def _collect_gradients(model: torch.nn.Module, grad_dict: dict, step_iter: int):
    """
    Official ZiCO-style gradient collection.

    Collect gradients only from:
        Conv2d weights
        Linear weights

    For each layer, store one flattened gradient vector per batch.
    """
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            if mod.weight.grad is None:
                continue

            grad = mod.weight.grad.detach().cpu().reshape(-1).numpy()

            if step_iter == 0:
                grad_dict[name] = [grad]
            else:
                if name not in grad_dict:
                    grad_dict[name] = [grad]
                else:
                    grad_dict[name].append(grad)

    return grad_dict


def _calculate_zico(grad_dict: dict):
    """
    Official ZiCO core formula.

    For each Conv/Linear layer:
        std = std(gradients across batches)
        mean_abs = mean(abs(gradients across batches))
        layer_score = log(sum(mean_abs / std))

    Final ZiCO:
        sum(layer_score over layers)
    """
    zico_score = 0.0

    for modname in grad_dict.keys():
        grad_array = np.array(grad_dict[modname])

        # ZiCO needs variation across more than one gradient sample.
        if grad_array.shape[0] < 2:
            continue

        nsr_std = np.std(grad_array, axis=0)
        nonzero_idx = np.nonzero(nsr_std)[0]

        if len(nonzero_idx) == 0:
            continue

        nsr_mean_abs = np.mean(np.abs(grad_array), axis=0)

        tmpsum = np.sum(
            nsr_mean_abs[nonzero_idx] / nsr_std[nonzero_idx]
        )

        if tmpsum > 0 and np.isfinite(tmpsum):
            zico_score += np.log(tmpsum)

    return float(zico_score)


@measure("zico", bn=True, mode="param")
def compute_zico_per_weight(
    net,
    medmnist_dataset,
    inputs,
    targets,
    loss_fn=None,
    split_data=1,
    **kwargs
):
    """
    ZiCO measure integrated into the repo's foresight/pruners/measures system.

    This preserves the official ZiCO logic:
        collect Conv/Linear gradients over multiple mini-batches
        calculate inverse coefficient-of-variation score

    Adapted for this repo:
        no hardcoded .cuda()
        handles NetworkCIFAR returning (logits, features)
        handles MedMNIST target shapes
    """
    if loss_fn is None:
        raise ValueError("ZiCO requires loss_fn.")

    device = inputs.device
    was_training = net.training
    net.train()

    grad_dict = {}

    # If predictive.py passes one large tensor, split it into multiple chunks.
    # ZiCO needs at least 2 gradient samples to compute std across samples.
    if split_data is None or split_data < 2:
        split_data = 2

    input_chunks = torch.chunk(inputs, split_data)
    target_chunks = torch.chunk(targets, split_data)

    step_iter = 0

    for input_chunk, target_chunk in zip(input_chunks, target_chunks):
        if input_chunk.shape[0] == 0:
            continue

        net.zero_grad(set_to_none=True)

        input_chunk = input_chunk.to(device)
        target_chunk = _prepare_targets(target_chunk, medmnist_dataset, device)

        outputs = net(input_chunk)
        logits = _get_logits(outputs)

        loss = loss_fn(logits, target_chunk)
        loss.backward()

        grad_dict = _collect_gradients(
            model=net,
            grad_dict=grad_dict,
            step_iter=step_iter
        )

        step_iter += 1

    zico_score = _calculate_zico(grad_dict)

    net.zero_grad(set_to_none=True)

    if was_training:
        net.train()
    else:
        net.eval()

    return [torch.tensor(zico_score)]