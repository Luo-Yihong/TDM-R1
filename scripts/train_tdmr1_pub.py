
from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
import shutil
import sys
import shlex
import subprocess
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_tdmr1 import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch.distributed as dist


def convert(x, s=7):
    return s * x / 1000 / (1 + (s-1) * x / 1000) * 1000

def inv_convert(y, s=7):
    return 1000 * y / (1000 * s - (s - 1) * y)

def generate_shared_sampled_timesteps(accelerator, train_timesteps, M):
    """
    跨 GPU 生成共享的采样 timesteps。
    
    Args:
        accelerator: Accelerator 对象
        train_timesteps: 可采样的 timestep 列表
        M: 要采样的数量
    
    Returns:
        sampled_timesteps: 列表，所有 GPU 上的值相同
    """
    device = accelerator.device
    
    if accelerator.is_main_process:
        sampled = random.sample(train_timesteps, M)
        sampled_tensor = torch.tensor(sampled, device=device, dtype=torch.long)
    else:
        sampled_tensor = torch.empty(M, device=device, dtype=torch.long)
    
    dist.broadcast(sampled_tensor, src=0)
    
    return sampled_tensor.tolist()

def generate_shared_timestep(accelerator, tmin = 0.5):
    """
    跨 GPU 生成一个共享的 timestep，均匀分布在 [tmin, 1]。
    
    Args:
        accelerator: Accelerator 对象
    
    Returns:
        timestep: 标量 tensor，所有 GPU 上的值相同
    """
    device = accelerator.device
    
    if accelerator.is_main_process:
        timestep = tmin + torch.rand(1, device=device) * (1 - tmin)
    else:
        timestep = torch.empty(1, device=device)
    
    dist.broadcast(timestep, src=0)
    
    return timestep

metrics_history = defaultdict(list)


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


def compute_group_dgpo_loss_allreduce(
    model_v, ref_old_v, target_v, advantages,
    group_info, accelerator, beta_dpo, group_size = 24, dsm_loss = None, ref_dsm_loss = None,
):
    """AllReduce实现的梯度等价版本"""
    batch_size = model_v.shape[0]
    device = model_v.device
    
    if dsm_loss is None:
        dsm_loss = (target_v - model_v).square().reshape(batch_size, -1).mean(dim=1)
    if ref_dsm_loss is None:
        with torch.no_grad():
            ref_dsm_loss = (target_v - ref_old_v).square().reshape(batch_size, -1).mean(dim=1)
    
    delta_diff = dsm_loss.detach() - ref_dsm_loss.detach()
    per_sample_term = advantages * beta_dpo * delta_diff / group_size
    
    local_group_indices = group_info['local_group_indices']
    num_groups = group_info['num_groups']
    
    local_group_sums = torch.zeros(num_groups, device=device, dtype=per_sample_term.dtype)
    local_group_sums.scatter_add_(0, local_group_indices, per_sample_term)
    
    global_group_sums = local_group_sums.clone().detach()
    dist.all_reduce(global_group_sums, op=dist.ReduceOp.SUM)
    
    # group_weights = 1 - torch.sigmoid(global_group_sums)
    group_weights = torch.sigmoid(global_group_sums)
    local_weights = group_weights[local_group_indices]
    
    loss = (local_weights.detach() * advantages * dsm_loss).mean()
    
    return loss, local_weights.detach().mean()

def precompute_group_info(prompt_ids, accelerator):
    """用 prompt_ids 预计算 group 信息（更快）"""
    batch_size = prompt_ids.shape[0]
    
    local_group_ids = prompt_ids.view(batch_size, -1)
    all_group_ids = accelerator.gather(local_group_ids)
    
    _, inverse_indices = torch.unique(all_group_ids, dim=0, return_inverse=True)
    num_groups = inverse_indices.max().item() + 1
    
    rank = accelerator.process_index
    start_idx = rank * batch_size
    end_idx = start_idx + batch_size
    
    local_group_indices = inverse_indices[start_idx:end_idx]
    
    return {
        'inverse_indices': inverse_indices,
        'local_group_indices': local_group_indices,
        'num_groups': num_groups,
        'local_start': start_idx,
        'local_end': end_idx,
        'batch_size': batch_size
    }

def generate_shared_noise_for_groups(x0, group_info, accelerator):
    """
    为每个 group 生成共享的噪声，跨 GPU 同步。
    
    Args:
        x0: [batch_size, C, H, W] 用于获取形状
        group_info: 预计算的 group 信息，包含 inverse_indices, num_groups 等
        accelerator: Accelerator 对象
    
    Returns:
        noise_diffuse: [batch_size, C, H, W] 每个 group 内共享相同的噪声
    """
    batch_size = x0.shape[0]
    device = x0.device
    num_groups = group_info['num_groups']
    inverse_indices = group_info['inverse_indices']
    
    # 1. 只在 rank 0 生成噪声，然后广播
    if accelerator.is_main_process:
        group_noises = torch.randn(num_groups, *x0.shape[1:], device=device)
    else:
        group_noises = torch.empty(num_groups, *x0.shape[1:], device=device)
    
    # 广播噪声到所有 GPU
    dist.broadcast(group_noises, src=0)
    
    # 2. 根据 inverse_indices 把噪声分配给每个样本
    all_noises = group_noises[inverse_indices]  # [total_batch_size, C, H, W]
    
    # 3. 提取当前 GPU 的噪声
    noise_diffuse = all_noises[group_info['local_start']:group_info['local_end']]
    
    return noise_diffuse

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0
    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(repeated_indices[start:end])
            yield per_card_samples[self.rank]
    # def __iter__(self):
    #     while True:
    #         # Generate a deterministic random sequence to ensure all replicas are synchronized
    #         g = torch.Generator()
    #         g.manual_seed(self.seed + self.epoch)
            
    #         # Randomly select m unique samples
    #         indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
    #         # Repeat each sample k times to generate n*b total samples
    #         repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
    #         # Shuffle to ensure uniform distribution
    #         shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
    #         shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
    #         # Split samples to each replica
    #         per_card_samples = []
    #         for i in range(self.num_replicas):
    #             start = i * self.batch_size
    #             end = start + self.batch_size
    #             per_card_samples.append(shuffled_samples[start:end])
            
    #         # Return current replica's sample indices
    #         yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def predict_v(transformer, noisy_samples, timesteps, embeds, pooled_embeds, config, cfg = True, uncond = False, cfg_scale = None, return_all = False):
    """
    修改后的函数：计算模型预测速度的对数概率。
    保持输入参数不变。
    """
    if cfg_scale is None:
        cfg_scale = config.sample.guidance_scale
    if config.train.cfg and uncond:
        embeds_uncond, embeds_cond = embeds.chunk(2)
        pooled_embeds_uncond, pooled_embeds_cond = pooled_embeds.chunk(2)
        predicted_velocity = transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds_uncond,
            pooled_projections=pooled_embeds_uncond,
            return_dict=False,
        )[0]
    elif config.train.cfg and cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([noisy_samples] * 2),
            timestep=torch.cat([timesteps] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond = noise_pred_uncond.detach()
        predicted_velocity = (
            noise_pred_uncond
            + cfg_scale
            * (noise_pred_text - noise_pred_uncond)
        )
        if return_all:
            return predicted_velocity, noise_pred_text, noise_pred_uncond
    elif config.train.cfg and (not cfg):
        embeds_uncond, embeds_cond = embeds.chunk(2)
        pooled_embeds_uncond, pooled_embeds_cond = pooled_embeds.chunk(2)
        predicted_velocity = transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds_cond,
            pooled_projections=pooled_embeds_cond,
            return_dict=False,
        )[0]
    else:
        predicted_velocity = transformer(
            hidden_states=noisy_samples,
            timestep=timesteps,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    return predicted_velocity


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        pipeline.transformer.set_adapter("tdm")
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                                    )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
            for key, value in {**{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()}}.items():
                metrics_history[key].append((global_step, value))

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_run_snapshot(run_root, config):
    """Save script copy, resolved config JSON, argv, optional --config source file, and git metadata."""
    run_snapshot_dir = os.path.join(run_root, "run_snapshot")
    os.makedirs(run_snapshot_dir, exist_ok=True)

    shutil.copy2(__file__, os.path.join(run_snapshot_dir, os.path.basename(__file__)))

    with open(os.path.join(run_snapshot_dir, "config_resolved.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)

    with open(os.path.join(run_snapshot_dir, "argv.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(a) for a in sys.argv))

    config_path = None
    for a in sys.argv:
        if a.startswith("--config="):
            config_path = a.split("=", 1)[1]
            break
    if config_path is None:
        for i, a in enumerate(sys.argv):
            if a == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                break
    if config_path:
        src = os.path.abspath(os.path.expanduser(config_path))
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(run_snapshot_dir, os.path.basename(src)))

    git_lines = []
    cur = os.path.dirname(os.path.abspath(__file__))
    git_root = None
    while cur != os.path.dirname(cur):
        if os.path.isdir(os.path.join(cur, ".git")):
            git_root = cur
            break
        cur = os.path.dirname(cur)

    if git_root:
        for args, title in (
            (["git", "-C", git_root, "rev-parse", "HEAD"], "HEAD"),
            (["git", "-C", git_root, "status", "-sb"], "status"),
        ):
            try:
                r = subprocess.run(
                    args, capture_output=True, text=True, check=False, timeout=60
                )
                out = (r.stdout or r.stderr or "").strip()
                git_lines.append(f"=== {title} ===\n{out}\n")
            except Exception as e:
                git_lines.append(f"=== {title} (failed) ===\n{e!s}\n")
    else:
        git_lines.append("No .git directory found walking up from the training script.\n")

    with open(os.path.join(run_snapshot_dir, "git_info.txt"), "w", encoding="utf-8") as f:
        f.write("".join(git_lines))


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    # num_train_timesteps = config.sample.num_steps - 1
    num_train_timesteps = config.num_train_timesteps
    assert config.trunc_steps >= num_train_timesteps
    
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        print(config)
        base_name = "tdm_r1-clip-neat-FixV"
        if config.use_ema_ref:
            base_name += "-emaref"
        else:
            base_name += "-froref"
        reward_parts = [
            k if float(v) == 1.0 else f"{k}{v}"
            for k, v in sorted(config.reward_fn.items())
            if float(v) != 0.0
        ]
        if reward_parts:
            base_name += f"-rwd_{'+'.join(reward_parts)}"
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        my_proj_name = f"{base_name}-G{config.sample.num_image_per_prompt}-{config.sample.num_steps}steps-beta{config.train.beta}-{config.train.beta_dpo}-BiasTmin{config.t_min_dgpo}-RwdTmin{config.t_min_dgpo_reward}-trunc{config.trunc_steps}-tdm_w{config.train.tdm_weight}_rlCFG{config.rl_cfg}"
        if config.use_tweight:
            my_proj_name += f"-t_w"
        my_proj_name += f"-RLbeta{config.train.rl_adam_beta1}"
        my_proj_name +=  "_" + unique_id
        if not config.sample.global_std:
            my_proj_name += "_Localstd"
        wandb.init(
            project="TDM-R1",
            name = my_proj_name,
            # mode="offline"  # 添加这行
        )
        save_run_snapshot(os.path.join(config.logdir, my_proj_name), config)
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", subfolder="scheduler")
    pipeline.scheduler.config['flow_shift'] = 3# the flow_shift can be changed from 1 to 6.
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="tdm")
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="fake")
        pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config, adapter_name="dgpo")
        
    
    pipeline.transformer.set_adapter("tdm")
    transformer = pipeline.transformer
    transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "tdm" in name:
            assert param.requires_grad == True
            transformer_trainable_parameters.append(param)

    pipeline.transformer.set_adapter("fake")
    fake_transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "fake" in name:
            assert param.requires_grad == True
            fake_transformer_trainable_parameters.append(param)

    pipeline.transformer.set_adapter("dgpo")
    dgpo_transformer_trainable_parameters = []
    for name, param in transformer.named_parameters():
        if "dgpo" in name:
            assert param.requires_grad == True
            dgpo_transformer_trainable_parameters.append(param)

    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=config.ema_decay_g, update_step_interval=config.ema_update_interval, device=accelerator.device)

    ema_old = EMAModuleWrapper(dgpo_transformer_trainable_parameters, decay=0, update_step_interval=1, device=accelerator.device)

    if config.train.lora_path:
        print(f"Loading from {config.train.lora_path}")
        ema.load(f"{config.train.lora_path}")
        ema.copy_ema_to(transformer_trainable_parameters)

    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate / 4,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    optimizer_fake = optimizer_cls(
        fake_transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    optimizer_dgpo = optimizer_cls(
        dgpo_transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.rl_adam_beta1, config.train.rl_adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )


    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            # num_workers=1,
            num_workers=0,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            # num_workers=8,
            num_workers=0,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, optimizer_fake, optimizer_dgpo, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, optimizer_fake, optimizer_dgpo, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    # while True:
    while config.max_epochs is None or epoch < config.max_epochs:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            pipeline.transformer.set_adapter("tdm")
            save_dir = os.path.join(config.logdir, my_proj_name)
            save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
            os.makedirs(save_root, exist_ok=True)
            save_ema_pth = os.path.join(save_root, "ema.ckpt")
            ema.save(save_ema_pth)
            save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)


        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            pipeline.transformer.set_adapter("tdm")
            with autocast():
                with torch.no_grad():
                    images, latents_list, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator,
                )

            latents = torch.stack(
                latents_list, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)
            zeros_t = torch.zeros(
                    config.sample.train_batch_size, 1, 
                    dtype=timesteps.dtype, 
                    device=timesteps.device
                )
            timesteps = torch.cat([timesteps, zeros_t], dim=1)  # (batch_size, num_steps + 1)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "x0": latents_list[-1],
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                num_samples_new = min(16, len(images))  # 建议用16，正好4x4网格
                sample_indices_new = random.sample(range(len(images)), num_samples_new)
                selected_images = torch.stack([images[i] for i in sample_indices_new])
                grid = make_grid(
                        selected_images, 
                        nrow=4,  # 每行4张图片
                        padding=2,  # 图片间距
                        normalize=True,  # 自动归一化到[0,1]
                        value_range=(0, 1)  # 如果你的图片已经在[0,1]范围内
                    )
                save_dir = os.path.join(config.logdir, my_proj_name)
                os.makedirs(save_dir, exist_ok=True)
                save_image(grid, os.path.join(save_dir, f"grid_epoch_{epoch}.jpg") )



                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        # The purpose of repeating `adv` along the timestep dimension here is to make it easier to introduce timestep-dependent advantages later, such as adding a KL reward.
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, config.sample.eval_num_steps)
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            reward_metrics = {f"reward_{key}": value.mean() for key, value in gathered_rewards.items() 
                     if '_strict_accuracy' not in key and '_accuracy' not in key}
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )
            for key, value in reward_metrics.items():
                metrics_history[key].append((global_step, float(value)))


        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            if config.use_bi:
                advantages = stat_tracker.update(prompts, gathered_rewards['avg'], type = 'grpo_bi')
            elif config.use_dars:
                advantages = stat_tracker.update(prompts, gathered_rewards['avg'], type = 'dars')
            else:
                advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
                metrics_history["zero_std_ratio"].append((global_step, zero_std_ratio))
                metrics_history["reward_std_mean"].append((global_step, reward_std_mean))
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        # del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        # samples = {k: v[mask] for k, v in samples.items()}
        samples = {k: v for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )
        assert num_timesteps == config.sample.num_steps + 1

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            # samples = {k: v[perm] for k, v in samples.items()}
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }
            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            M = num_train_timesteps
            train_timesteps = [step_index  for step_index in range(config.sample.num_steps)]
            # t_min_dgpo = 550
            t_min_dgpo = config.t_min_dgpo

            def convert(x, s=7):
                return s * x / 1000 / (1 + (s-1) * x / 1000) * 1000

            def inv_convert(y, s=7):
                return 1000 * y / (1000 * s - (s - 1) * y)

            
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                group_info = precompute_group_info(sample["prompt_ids"], accelerator)
                tau_list = []
                tau_list_dgpo = []

                N = len(train_timesteps)
                device = sample["timesteps"].device

                j_dgpo_list = []
                tau_list_dgpo_train = []
                weights = torch.arange(N, device=device, dtype=torch.float32) + 1 
                probs = weights / weights.sum() # you can try different sampling strategies.
                # p_clean = getattr(config, 'dgpo_clean_p', 0.5)
                # probs = torch.full((N,), (1.0 - p_clean) / N, device=device, dtype=torch.float32)
                # probs[-1] = p_clean + (1.0 - p_clean) / N # you can try different sampling strategies.
                # Make the tau_tmp_inv_dgpo_train is close to uniformly sampled from [inv_convert(tmin), 1000]
                t_min_dgpo_reward = config.t_min_dgpo_reward
                for j_idx in range(len(train_timesteps)):
                    # ===== DGPO: 独立采样 j_dgpo =====
                    if accelerator.is_main_process:
                        j_dgpo_tmp = torch.multinomial(probs, num_samples=1)  # [1]
                    else:
                        j_dgpo_tmp = torch.zeros(1, dtype=torch.long, device=device)
                    torch.distributed.broadcast(j_dgpo_tmp, src=0)
                    j_dgpo_tmp = j_dgpo_tmp.item()
                    j_dgpo_list.append(j_dgpo_tmp)

                    t_next_dgpo = sample["timesteps"][:, j_dgpo_tmp + 1]
                    t_min_tensor = torch.tensor(t_min_dgpo_reward, device=device, dtype=t_next_dgpo.dtype)
                    lower_bound = torch.maximum(t_next_dgpo, t_min_tensor.expand_as(t_next_dgpo))
                    lower_bound_inv = inv_convert(lower_bound, s=3)
                    tau_scalar_dgpo = generate_shared_timestep(accelerator, tmin=0.0)
                    tau_tmp_inv_dgpo_train = tau_scalar_dgpo * (1000 - (lower_bound_inv + 20)) + (lower_bound_inv + 20)
                    tau_tmp_dgpo_train = convert(tau_tmp_inv_dgpo_train, s=3)
                    tau_list_dgpo_train.append(tau_tmp_dgpo_train)

                # 在原始空间均匀采样 tau_dgpo ∈ [lower_bound, t_curr]
                for j_idx in range(len(train_timesteps)):
                    t_next_tmp = sample["timesteps"][:, j_idx+1]  # [0, 1000]
                    
                    # 原始 tau（用于 fake 和 tdm）
                    tau_scalar = generate_shared_timestep(accelerator, tmin=0.0)
                    t_next_tmp_inv = inv_convert(t_next_tmp, s=3)
                    tau_tmp_inv = tau_scalar * (1000 - (t_next_tmp_inv + 20)) + (t_next_tmp_inv + 20)
                    tau_tmp = convert(tau_tmp_inv, s=3)
                    tau_list.append(tau_tmp)
                    
                    # DGPO 的 tau（有 t_min 约束，且 tau_dgpo >= t_next）
                    tau_scalar_dgpo = generate_shared_timestep(accelerator, tmin=0.0)
                    t_min_tensor = torch.tensor(t_min_dgpo, device=t_next_tmp.device, dtype=t_next_tmp.dtype)
                    # 下限是 max(t_next, t_min_dgpo)
                    lower_bound = torch.maximum(t_next_tmp, t_min_tensor.expand_as(t_next_tmp))
                    lower_bound_inv = inv_convert(lower_bound, s=3)
                    # 在 inv 空间中采样，下限是 lower_bound_inv + 20
                    tau_tmp_inv_dgpo = tau_scalar_dgpo * (1000 - (lower_bound_inv + 20)) + (lower_bound_inv + 20)
                    tau_tmp_dgpo = convert(tau_tmp_inv_dgpo, s=3)
                    # 再次确保 tau_dgpo >= max(t_next, t_min_dgpo)
                    tau_tmp_dgpo = torch.maximum(tau_tmp_dgpo, lower_bound)
                    tau_list_dgpo.append(tau_tmp_dgpo)


                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                M = num_train_timesteps
                sampled_timesteps = generate_shared_sampled_timesteps(accelerator, train_timesteps, M)
                trunc_steps = config.trunc_steps
                for j in tqdm(
                    sampled_timesteps,
                    total=M, 
                    desc="Sampling Timesteps",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        x0 = sample["x0"]
                        t = sample["timesteps"][:, j]
                        t_next = sample["timesteps"][:, j+1] # [0, 1000]

                        tau = tau_list[j]
                        tau_dgpo = tau_list_dgpo[j]

                        sigmas_t = (t / 1000).reshape(x0.shape[0],1,1,1) 
                        sigmas_tau = (tau / 1000).reshape(x0.shape[0],1,1,1) 
                        sigmas_tau_dgpo = (tau_dgpo / 1000).reshape(x0.shape[0], 1, 1, 1)
                        sigmas_tnext = (t_next / 1000).reshape(x0.shape[0],1,1,1) 

                        j_dgpo = j_dgpo_list[j]
                        t_dgpo = sample["timesteps"][:, j_dgpo]
                        t_next_dgpo = sample["timesteps"][:, j_dgpo+1]
                        tau_dgpo_train = tau_list_dgpo_train[j]

                        sigmas_tau_dgpo_train = (tau_dgpo_train / 1000).reshape(x0.shape[0],1,1,1)
                        sigmas_tnext_dgpo = (t_next_dgpo / 1000).reshape(x0.shape[0],1,1,1)
                        xt_dgpo = sample["latents"][:, j_dgpo]
                        xnext_dgpo = sample["next_latents"][:, j_dgpo]



                        xt = sample["latents"][:, j]
                        # ============ TDM-Style Fake Score Training ============
                        with autocast():
                            pipeline.transformer.set_adapter("tdm")
                            with torch.no_grad():
                                model_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg=False)
                                model_x0 = xt - sigmas_t * model_v
                                model_xnext = xt - (sigmas_t - sigmas_tnext) * model_v

                        alpha = (1 - sigmas_tau) / (1 - sigmas_tnext)
                        beta = (sigmas_tau ** 2 - (sigmas_tnext * alpha) ** 2) ** 0.5
                        noise_diffuse = torch.randn_like(x0)
                        xtau = model_xnext * alpha + beta * noise_diffuse
                        target_v = (xtau - model_x0) / sigmas_tau

                        with autocast():
                            with torch.no_grad():
                                with transformer.module.disable_adapter():
                                    real_v = predict_v(transformer, xtau, tau, embeds, pooled_embeds, config, cfg=False)

                        pipeline.transformer.set_adapter("fake")
                        with autocast():
                            fake_v = predict_v(transformer, xtau, tau, embeds, pooled_embeds, config, cfg=False)
                        loss_fake = (target_v.detach() - fake_v).square().mean()
                        loss_reg_fake = (real_v - fake_v).square().mean()

                        accelerator.backward(loss_fake + 0.02 * loss_reg_fake)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer_fake.step()
                        optimizer_fake.zero_grad()

                        # ============ DGPO-style Training of Surrogate Reward ============
                        # 为 DGPO 重新计算 alpha_dgpo, beta_dgpo, xtau_dgpo
                        alpha_dgpo_train = (1 - sigmas_tau_dgpo_train) / (1 - sigmas_tnext_dgpo)
                        beta_dgpo_train = (sigmas_tau_dgpo_train ** 2 - (sigmas_tnext_dgpo * alpha_dgpo_train) ** 2) ** 0.5
                        noise_diffuse_dgpo = generate_shared_noise_for_groups(x0, group_info, accelerator)
                        xtau_dgpo_train = xnext_dgpo * alpha_dgpo_train + beta_dgpo_train * noise_diffuse_dgpo
                        # Let eta = 0, and gap between xt and xt-1 tends to 0, combine eq.x, we obtain the following target:
                        target_v_dgpo = - xnext_dgpo / (1 - sigmas_tnext_dgpo) + noise_diffuse_dgpo / beta_dgpo_train * (sigmas_tau_dgpo_train + sigmas_tnext_dgpo ** 2 * alpha_dgpo_train / (1 - sigmas_tnext_dgpo))  
                        with autocast():
                            with torch.no_grad():
                                with transformer.module.disable_adapter():
                                    real_v_dgpo = predict_v(transformer, xtau_dgpo_train, tau_dgpo_train, embeds, pooled_embeds, config, cfg=False)
                        advantages = sample["advantages"][:, j]
                        advantages = advantages.clip(min = -5, max = 5)
                        pipeline.transformer.set_adapter("dgpo")
                        with autocast():
                            with torch.no_grad():
                                ema_old.copy_ema_to(dgpo_transformer_trainable_parameters, store_temp=True)
                                old_v = predict_v(transformer, xtau_dgpo_train, tau_dgpo_train, embeds, pooled_embeds, config, cfg = False)
                                ema_old.copy_temp_to(dgpo_transformer_trainable_parameters)
                            dgpo_v = predict_v(transformer, xtau_dgpo_train, tau_dgpo_train, embeds, pooled_embeds, config, cfg=False)
                        dgpo_dsm_loss = 1 * (target_v_dgpo - dgpo_v).square().reshape(x0.shape[0],-1).mean(dim=1)
                        old_dgpo_dsm_loss = 1 * (target_v_dgpo - old_v).square().reshape(x0.shape[0],-1).mean(dim=1)
                        differ_loss = ((dgpo_v - old_v).square().reshape(x0.shape[0],-1).mean(dim=1) )
                        differ_loss = differ_loss.mean().detach()
                        ratio = torch.exp(-dgpo_dsm_loss + old_dgpo_dsm_loss)
                        clip_range = config.clip_range
                        should_clip = torch.where(
                            advantages > 0,
                            ratio > 1.0 + clip_range,  # 好图：降噪变好太多，clip
                            ratio < 1.0 - clip_range,  # 差图：降噪变差太多，clip
                        )

                        dgpo_dsm_loss_clipped = torch.where(
                            should_clip,
                            dgpo_dsm_loss.detach(),
                            dgpo_dsm_loss,
                        )
                        dgpo_dsm_loss = dgpo_dsm_loss_clipped

                        loss_reg_dgpo = (dgpo_v.float() - real_v_dgpo.detach().float()).square().reshape(x0.shape[0], -1).mean(dim=1)
                        ref_dsm_loss = (target_v_dgpo.float() - real_v_dgpo.detach().float()).square().reshape(x0.shape[0], -1).mean(dim=1)
                        loss_reg_dgpo = loss_reg_dgpo.mean()

                        reg_threshold = getattr(config, "reg_threshold", 0.05)
                        ref_ratio = torch.exp(-dgpo_dsm_loss + ref_dsm_loss)
                        should_clip_ref = (loss_reg_dgpo > reg_threshold) & torch.where(
                            advantages > 0,
                            ref_ratio > 1.0,    # 好图：降噪变好，但偏离 frozen 太远，clip
                            ref_ratio < 1.0,    # 差图：降噪变差，但偏离 frozen 太远，clip
                        )
                        dgpo_dsm_loss = torch.where(
                            should_clip_ref,
                            dgpo_dsm_loss.detach(),
                            dgpo_dsm_loss,
                        )

                        ref_v = real_v_dgpo
                        ref_dsm_loss = 1 * (target_v_dgpo - ref_v).square().reshape(x0.shape[0],-1).mean(dim=1)

                        dgpo_loss, scale_term = compute_group_dgpo_loss_allreduce(
                            dgpo_v, ref_v, target_v_dgpo, advantages,
                            group_info, accelerator, config.train.beta_dpo, group_size=config.sample.num_image_per_prompt, dsm_loss = dgpo_dsm_loss, ref_dsm_loss = ref_dsm_loss
                        )

                        t_weight = 1
                        if config.use_tweight:
                            t_weight = (j_dgpo + 1) / config.sample.num_steps
                        accelerator.backward(t_weight * (dgpo_loss.mean() + config.train.beta * loss_reg_dgpo))
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer_dgpo.step()
                        optimizer_dgpo.zero_grad()
                        if accelerator.sync_gradients:
                            ema_old_decay = min(config.ema_old_decay_max, config.ema_old_decay_min + 0.001 * global_step)
                            ema_old.step(dgpo_transformer_trainable_parameters, global_step, decay = ema_old_decay)

                        # ============ TDM-R1 Training of few-step generator ============
                        with autocast():
                            pipeline.transformer.set_adapter("tdm")
                            model_v = predict_v(transformer, xt, t, embeds, pooled_embeds, config, cfg=False)
                            model_x0 = xt - sigmas_t * model_v
                            model_eps = xt + (1 - sigmas_t) * model_v

                        model_xnext = xt - (sigmas_t - sigmas_tnext) * model_v
                        noise_diffuse = torch.randn_like(x0)
                        noise_diffuse_dgpo = generate_shared_noise_for_groups(x0, group_info, accelerator)
                        
                        alpha_dgpo = (1 - sigmas_tau_dgpo) / (1 - sigmas_tnext)
                        beta_dgpo  = (sigmas_tau_dgpo ** 2 - (sigmas_tnext * alpha_dgpo) ** 2) ** 0.5
                        xtau = model_xnext * alpha + beta * noise_diffuse
                        xtau_dgpo = model_xnext * alpha_dgpo + beta_dgpo * noise_diffuse_dgpo

                        with autocast():
                            with torch.no_grad():
                                # fake 使用原始 tau
                                pipeline.transformer.set_adapter("fake")
                                fake_v = predict_v(transformer, xtau, tau, embeds, pooled_embeds, config, cfg=False)
                                with transformer.module.disable_adapter():
                                    real_v_for_fake, real_cond_v, real_uncond_v = predict_v(transformer, xtau, tau, embeds, pooled_embeds, config, cfg=True, cfg_scale=4.5, return_all = True)

                            # fake 相关；使用原始 tau
                            fake_x0 = xtau - fake_v * sigmas_tau
                            real_x0_for_fake = xtau - real_v_for_fake * sigmas_tau
                            real_cond_x0 = xtau - real_cond_v * sigmas_tau
                            real_uncond_x0 = xtau - real_uncond_v * sigmas_tau
                            
                            # dgpo 相关；使用 tau_dgpo
                            if j < trunc_steps:
                                # dgpo 使用 tau_dgpo
                                with torch.no_grad():
                                    pipeline.transformer.set_adapter("dgpo")
                                    dgpo_v_cfg, dgpo_cond_v, dgpo_uncond_v = predict_v(transformer, xtau_dgpo, tau_dgpo, embeds, pooled_embeds, config, cfg=True, cfg_scale=config.rl_cfg, return_all = True)
                                    with transformer.module.disable_adapter():
                                        dgpo_ref_v = predict_v(transformer, xtau_dgpo, tau_dgpo, embeds, pooled_embeds, config, cfg=False)
                                dgpo_x0 = xtau_dgpo - dgpo_v_cfg * sigmas_tau_dgpo
                                dgpo_cond_x0 = xtau_dgpo - dgpo_cond_v * sigmas_tau_dgpo
                                dgpo_uncond_x0 = xtau_dgpo - dgpo_uncond_v * sigmas_tau_dgpo
                                dgpo_ref_x0 = xtau_dgpo - dgpo_ref_v * sigmas_tau_dgpo
                                dgpo_revised_x0 = (model_x0 + dgpo_x0 - dgpo_ref_x0).detach()

                            cfg_revised_x0 = (model_x0 + 3.5 * (real_cond_x0 - real_uncond_x0)).detach()
                            kl_revised_x0 = (model_x0 + 1 * (real_cond_x0 - fake_x0)).detach()
                            revised_x0 = (model_x0 + real_x0_for_fake - fake_x0).detach()

                        weighting_factor = torch.abs(model_x0.double() - real_x0_for_fake.double() ).mean(dim=[1, 2, 3], keepdim=True).detach()

                        pipeline.transformer.set_adapter("tdm")
                        
                        err_kl = (kl_revised_x0 - model_x0).square()
                        err_cfg_reward = (cfg_revised_x0 - model_x0).square() # We regard cfg as a reward following JDM and DI++.
                        err_tdm = err_cfg_reward + err_kl
                        if config.use_huber:
                            huber_c = 1e-3
                            loss_tdm = torch.mean((torch.sqrt(err_tdm + huber_c**2) - huber_c) / weighting_factor)
                        else:
                            loss_tdm = (err_tdm / weighting_factor).mean()
                        loss_cfg = ((cfg_revised_x0.detach().double() - model_x0.double()).square() / weighting_factor).mean()

                        if config.use_huber:
                            loss_cfg_reward = 3.5 * torch.mean((torch.sqrt( err_cfg_reward + huber_c**2) - huber_c) / weighting_factor)
                            loss_kl = torch.mean((torch.sqrt( err_kl + huber_c**2) - huber_c) / weighting_factor)
                        else:
                            loss_cfg_reward = (err_cfg_reward / weighting_factor).mean()
                            loss_kl = (err_kl / weighting_factor).mean()

                        if j < trunc_steps:
                            weighting_dgpo = torch.abs(model_x0.double() - dgpo_x0.double() ).mean(dim=[1, 2, 3], keepdim=True).detach()
                            err_reward = (dgpo_revised_x0 - model_x0).square()
                            err_rl = err_reward + err_kl
                            if config.use_huber:
                                loss_reward = torch.mean((torch.sqrt( err_reward + huber_c**2) - huber_c) / weighting_dgpo)
                            else:
                                loss_reward = (err_reward / weighting_dgpo).mean()
                            loss = config.train.tdm_weight * loss_cfg_reward + (1 - config.train.tdm_weight) * loss_reward + loss_kl
                        else:
                            loss = config.train.tdm_weight * loss_tdm # drop the RL-related component

                        info["weighting_factor"].append(weighting_factor.mean().detach())
                        info["fake_loss"].append(loss_fake.detach())
                        info["tdmr1_tdm_loss"].append(loss_tdm.detach())
                        info["tdmr1_cfg_loss"].append(loss_cfg.detach())
                        info["tdmr1_kl_loss"].append(loss_kl.detach())
                        info["loss"].append(loss.detach())
                        info["scale_term"].append(scale_term.mean().detach())
                        info["dgpo_dsm_loss"].append(dgpo_dsm_loss.mean().detach())
                        info["dgpo_loss"].append(dgpo_loss.mean().detach())
                        info["dgpo_reg_loss"].append(loss_reg_dgpo.detach())
                        info["dgpo_differ_loss"].append(differ_loss.detach())
                        clip_ratio_total = should_clip.float().mean()
                        info["clip_ratio"].append(clip_ratio_total.detach())
                        info["clip_ref_ratio"].append(should_clip_ref.float().mean().detach())


                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                        if accelerator.sync_gradients:
                            if config.train.ema:
                                ema.step(transformer_trainable_parameters, global_step)

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:

                            with torch.no_grad():
                                images_4step = pipeline.vae.decode(x0[:4].to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor, return_dict=False)[0].clamp(-1,1)*0.5+0.5
                            save_image(images_4step, os.path.join(save_dir, f'x0_samples.jpg'), normalize = False, nrow = 2)

                            wandb.log(info, step=global_step)
                            
                            # 更新 metrics_history
                            for key, value in info.items():
                                if key not in ["epoch", "inner_epoch"]:
                                    metrics_history[key].append((global_step, float(value)))
                            
                            # 保存图片
                            save_dir = os.path.join(config.logdir, my_proj_name)
                            os.makedirs(save_dir, exist_ok=True)
                            print(save_dir)
                            
                            for metric_name, values in metrics_history.items():
                                if len(values) >= 1:
                                    steps, metric_values = zip(*values)
                                    plt.figure(figsize=(10, 6))
                                    # plt.plot(steps, metric_values)
                                    plt.plot(steps, metric_values, marker='o', markersize=8, linestyle='-' if len(values) > 1 else 'None')
                                    plt.title(f'{metric_name} over time')
                                    plt.xlabel('Global Step')
                                    plt.ylabel(metric_name)
                                    plt.grid(True)
                                    plt.savefig(os.path.join(save_dir, f'{metric_name}.jpg'), 
                                            dpi=150, bbox_inches='tight')
                                    plt.close()
                        global_step += 1
                        info = defaultdict(list)
        
        epoch+=1
    executor.shutdown(wait=False)
    
    accelerator.wait_for_everyone()
    
    accelerator.end_training()

    if accelerator.is_main_process:
        wandb.finish()
    
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()
    
    import multiprocessing as mp
    mp.active_children()
    
    import sys
    os._exit(0)  


if __name__ == "__main__":
    app.run(main)