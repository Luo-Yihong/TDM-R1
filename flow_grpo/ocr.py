from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union
from PIL import Image
import os
import fcntl
from contextlib import contextmanager


@contextmanager
def _paddleocr_init_lock():
    # Serialize PaddleOCR model download across processes that share the same
    # cache dir (default ~/.paddleocr). Without this, multiple ranks race on
    # downloading the same <model>.tar.tmp and one rank's shutil.move wipes the
    # .tmp file that another rank is about to move -> FileNotFoundError.
    lock_path = os.path.expanduser("~/.paddleocr.init.lock")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

# from torch.utils.data import Dataset, DataLoader, Sampler
# class TextPromptDataset(Dataset):
#     def __init__(self, dataset = "../grpo_flow/dataset/ocr", split='train'):
#         self.file_path = os.path.join(dataset, f'{split}.txt')
#         with open(self.file_path, 'r') as f:
#             self.prompts = [line.strip() for line in f.readlines()]
        
#     def __len__(self):
#         return len(self.prompts)
    
#     def __getitem__(self, idx):
#         return {"prompt": self.prompts[idx], "metadata": {}}

#     @staticmethod
#     def collate_fn(examples):
#         prompts = [example["prompt"] for example in examples]
#         metadatas = [example["metadata"] for example in examples]
#         return prompts, metadatas

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            lang="en",
            use_gpu=use_gpu,
            show_log=False
        )
        # # Run CPU PaddleOCR without MKLDNN and with a single CPU thread.
        # # Multiple ranks on the same node otherwise corrupt MKLDNN's primitive
        # # cache, which surfaces as either "could not create a primitive
        # # descriptor" or ReshapeOp receiving garbage negative dimensions.
        # if not use_gpu:
        #     os.environ.setdefault("FLAGS_use_mkldnn", "0")
        #     os.environ.setdefault("OMP_NUM_THREADS", "1")
        #     os.environ.setdefault("MKL_NUM_THREADS", "1")
        # with _paddleocr_init_lock():
        #     self.ocr = PaddleOCR(
        #         use_angle_cls=False,
        #         lang="en",
        #         use_gpu=use_gpu,
        #         enable_mkldnn=False,
        #         cpu_threads=1,
        #         show_log=False,
        #     )

    @torch.no_grad()
    def __call__(self, 
                images: Union[List[Image.Image], List[np.ndarray]], 
                prompts: List[str]) -> torch.Tensor:
        """
        Calculate OCR reward
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list
        :return: Reward tensor (CPU)
        """
        prompts = [prompt.split('"')[1] for prompt in prompts]
        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"
        for img, prompt in zip(images, prompts):
            # Convert image format
            if isinstance(img, Image.Image):
                img = np.array(img)
            
            try:
                # OCR recognition
                result = self.ocr.ocr(img, cls=False)
                # Extract recognized text (handle possible multi-line results)
                recognized_text = ''.join([res[1][0] if res[1][1] > 0 else '' for res in result[0]]) if result[0] else ''
                
                recognized_text = recognized_text.replace(' ', '').lower()
                prompt = prompt.replace(' ', '').lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = distance(recognized_text, prompt)
                # Recognized many unrelated characters, only add one character penalty
                if dist > len(prompt):
                    dist = len(prompt)
                
            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                print(f"OCR processing failed: {str(e)}")
                dist = len(prompt)  # Maximum penalty
            reward = 1-dist/(len(prompt))
            rewards.append(reward)

        return rewards

# def ocr_score(device):
#     from ocr import OcrScorer
#     scorer = OcrScorer()
#     def _fn(images, prompts, metadata):
#         if isinstance(images, torch.Tensor):
#             images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
#             images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
#         scores = scorer(images, prompts)
#         # change tensor to list
#         return scores, {}
#     return _fn

if __name__ == "__main__":
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)
    example_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'
    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Call scorer and print result
    reward = scorer([example_image], [example_prompt])
    print(f"OCR Reward: {reward}")