import ml_collections
import os
from huggingface_hub import hf_hub_download

TDM_LORA_PATH = hf_hub_download(
    repo_id="Luo-Yihong/TDM_sd3-5_lora",
    filename="tdm_sd3-5_lora.ckpt",
)

def _load_source(name, path):
    try:
        import imp
        return imp.load_source(name, path)
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

base = _load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "/root/models/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")
    config.use_lora = True

    # Sampling (4-step TDM, 24 imgs / prompt)
    config.sample.num_steps = 4
    config.sample.eval_num_steps = 4
    config.sample.guidance_scale = 1
    config.sample.num_image_per_prompt = 24
    config.sample.noise_level = 0
    config.sample.global_std = True
    config.sample.same_latent = False

    config.resolution = 512

    # Training
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.adam_beta1 = 0.
    config.train.rl_adam_beta1 = 0.9
    config.train.rl_adam_beta2 = 0.999
    config.train.beta = 0.001
    config.train.beta_dpo = 1
    config.train.tdm_weight = 0.3
    config.train.ema = True
    config.train.lora_path = TDM_LORA_PATH

    # Schedule
    config.save_freq = 60
    config.eval_freq = 60
    config.max_epochs = 10000

    # TDM-specific
    config.num_train_timesteps = 2
    config.ema_update_interval = 1
    config.ema_decay_g = 0.9
    config.ema_old_decay_min = 0.001
    config.ema_old_decay_max = 0.3
    config.use_ema_ref = False
    config.use_tweight = True
    config.use_huber = False
    config.use_bi = False
    config.use_dars = False
    config.reg_threshold = 0.05
    config.clip_range = 1e-3
    config.rl_cfg = 4.5
    config.t_min_dgpo_reward = 550

    # Prompts / rewards
    config.prompt_fn = "general_ocr"
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_8gpu_G24_4step():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    config.sample.train_batch_size = 12
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 8 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch

    config.t_min_dgpo = 300
    config.t_min_dgpo_reward = 300
    config.train.beta_dpo = 10
    config.trunc_steps = 4

    config.reward_fn = {
        "ocr": 1.0,
    }
    return config

def geneval_sd3_8gpu_G24_24_4step():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    config.prompt_fn = "geneval"

    config.sample.train_batch_size = 6
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch

    config.t_min_dgpo = 250
    config.rl_cfg = 2.5
    config.trunc_steps = 4

    config.reward_fn = {
        "geneval": 1.0,
    }
    return config
    
def imagereward_sd3_8gpu_G24():
    gpu_number = 8
    config = compressibility()

    config.sample.train_batch_size = 12
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch

    config.t_min_dgpo = 400
    config.rl_cfg = 3.5
    config.clip_range = 2e-3
    config.trunc_steps = 2

    config.reward_fn = {
        "imagereward": 1.0,
    }
    return config


def hps_sd3_8gpu_G24():
    gpu_number = 8
    config = compressibility()

    config.sample.train_batch_size = 12
    config.sample.num_batches_per_epoch = int(24/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch

    config.t_min_dgpo = 400
    config.rl_cfg = 3.5
    config.train.beta_dpo = 10
    config.clip_range = 2e-3
    config.trunc_steps = 2

    config.reward_fn = {
        "hpsv2": 1.0,
    }
    return config

def get_config(name):
    return globals()[name]()
