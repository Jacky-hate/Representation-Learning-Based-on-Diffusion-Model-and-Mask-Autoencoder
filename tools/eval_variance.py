import argparse
import os
import lpips
import cv2
from utils import calculate_psnr, calculate_ssim
from tqdm import tqdm
import numpy as np

def eval(dir):
	imgs = []
	for file in tqdm(sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]))):
		# Load images
		img = lpips.load_image(os.path.join(dir,file))
		#cv2 to numpy
		img = np.array(img)
		imgs.append(img)
	imgs = np.array(imgs) / 255.0 * 2.0 - 1.0
	print(imgs.shape, imgs.min(), imgs.max())
	#flatten (B, H, W, C) to (B, H*W*C)
	imgs = imgs.reshape(imgs.shape[0], -1)
	#calculate variance of each pixel across all images and get the mean
	var = np.var(imgs, axis=0).mean()
	print(f'Variance: {var}')
	return var

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d','--dir', type=str, default='/mnt/urchin/kzou/code/transformer/output_sample/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/val_one_ep500_1024')
	parser.add_argument('-o','--out', type=str, default='./example_dists.txt')
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

	opt = parser.parse_args()
	eval(opt.dir)
