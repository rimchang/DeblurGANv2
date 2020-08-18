import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator


class Predictor:
    def __init__(self, weights_path: str, model_name: str = ''):
        with open('config/config_RealBlurJ_bsd_gopro_pretrain_ragan-ls.yaml') as cfg:
            config = yaml.load(cfg)
        model = get_generator(model_name or config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        if mask is None:
            mask = np.ones_like(x, dtype=np.float32)
        else:
            mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 #* 255.0
        return x

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)
        with torch.no_grad():
            inputs = [img.cuda()]
            if not ignore_mask:
                inputs += [mask]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


def main(img_pattern: str,
         mask_pattern: Optional[str] = None,
         weights_path='provided_model/fpn_inception.h5',
         out_dir='result/',
         side_by_side: bool = False):
    def sorted_glob(pattern):
        return sorted(glob(pattern))

    if '.txt' not in img_pattern:
        imgs = sorted_glob(img_pattern)
        mask_pattern = None
        masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
        pairs = list(zip(imgs, masks))
        names = sorted([os.path.basename(x) for x in glob(img_pattern)])
        predictor = Predictor(weights_path=weights_path)
    else:
        imgs = open(img_pattern, 'rt').read().splitlines()
        imgs = list(map(lambda x: x.strip().split(' '), imgs))
        imgs = [blur_p for gt_p, blur_p in imgs]
        mask_pattern = None
        masks = sorted_glob(mask_pattern) if mask_pattern is not None else [None for _ in imgs]
        pairs = list(zip(imgs, masks))
        if 'kohler' not in img_pattern:
            names = [x.split('/') for x in imgs]
            names = [x[1] + "_" + x[-1] for x in names]
        else:
            names = [os.path.basename(x) for x in imgs]

        predictor = Predictor(weights_path=weights_path)
    os.makedirs(out_dir, exist_ok=True)
    for name, pair in tqdm(zip(names, pairs), total=len(names)):
        f_img, f_mask = pair
        f_img = os.path.join('dataset', f_img)
        img, mask = map(cv2.imread, (f_img, f_mask))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pred = predictor(img, mask)
        if side_by_side:
            pred = np.hstack((img, pred))
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

        out = pred
        srgb_out = np.power(out, 1 / 2.2)
        srgb_out = np.clip(srgb_out * 255, 0, 255) + 0.5
        srgb_out = srgb_out.astype('uint8')

        out = np.clip(out * 255, 0, 255) + 0.5
        out = out.astype('uint8')


        name = name.replace('jpg', 'png')
        cv2.imwrite(os.path.join(out_dir, name),
                    out)
        #cv2.imwrite(os.path.join(out_dir, 'srgb_'+name),
        #            srgb_out)

if __name__ == '__main__':
    Fire(main)
