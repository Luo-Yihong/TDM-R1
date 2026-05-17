# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union
import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    return_beta = False
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level
    
    # our sde
    prev_sample_mean = sample*(1+std_dev_t**2/(2*sigma)*dt)+model_output*(1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    
    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * torch.sqrt(-1*dt) * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1*dt))**2))
        - torch.log(std_dev_t * torch.sqrt(-1*dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_beta:
        return prev_sample, log_prob, prev_sample_mean, std_dev_t, (1+std_dev_t**2*(1-sigma)/(2*sigma))*dt
    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def eta_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output=model_output.float()
    sample=sample.float()
    if prev_sample is not None:
        prev_sample=prev_sample.float()

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step+1 for step in step_index]
    sigma = self.sigmas[step_index].to(sample.device).view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = self.sigmas[prev_step_index].to(sample.device).view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_max = self.sigmas[1].item()
    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))*noise_level

    # print(sigma.shape, sample.shape)

    sigma_2d = sigma.reshape(-1,1,1,1)
    sigma_prev_2d = sigma_prev.reshape(-1,1,1,1)
    if sigma_2d.shape[0] < sample.shape[0]:
        sigma_2d = sigma_2d.repeat(sample.shape[0],1,1,1)
        sigma_prev_2d = sigma_prev_2d.repeat(sample.shape[0],1,1,1)

    pred_samples = sample - sigma_2d * model_output
    pred_eps = sample + (1 - sigma_2d) * model_output

    eta = noise_level
    add_eps_mean = ((1 - eta ** 2) ** 0.5) * pred_eps
    prev_sample_mean = (1.0 - sigma_prev_2d) * pred_samples + sigma_prev_2d * add_eps_mean

    if prev_sample is None:
        variance_noise = torch.randn_like(pred_eps)
        add_eps = ((1 - eta ** 2) ** 0.5) * pred_eps + eta * variance_noise
        prev_sample =  (1.0 - sigma_prev_2d) * pred_samples + sigma_prev_2d * add_eps

    std_dev_t = std_dev_t * 0 + eta

    if eta > 0:
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((eta)**2))
            - torch.log(torch.as_tensor(eta))
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        # mean along all but batch dimension
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    else:
        # eta=0 -> deterministic step; log-density is degenerate. Return a
        # placeholder mean so downstream code that just stacks `log_probs`
        # (without using them in any loss) keeps working.
        log_prob = prev_sample.detach().reshape(prev_sample.shape[0], -1).mean(dim=1)

    return prev_sample, log_prob, prev_sample_mean, std_dev_t