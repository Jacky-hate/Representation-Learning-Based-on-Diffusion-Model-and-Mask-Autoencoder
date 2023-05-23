import argparse
import os
import lpips
import cv2
from utils import calculate_psnr, calculate_ssim
from tqdm import tqdm

def eval(dir, out='./example_dists.txt', use_gpu=True, version=0.1, max=128):
    
	## Initializing the model
	loss_fn = lpips.LPIPS(net='alex',version=version)
	if(use_gpu):
		loss_fn.cuda()

	# crawl directories
	f = open(out,'w')
	idx = 0
	dist = 0.0
	ssim = 0.0
	psnr = 0.0
	for file in tqdm(sorted(os.listdir("/mnt/urchin/kzou/code/transformer/output_sample/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/gt_val"), key=lambda x: int(x.split('.')[0]))):
		# Load images
		img0 = lpips.load_image(os.path.join("/mnt/urchin/kzou/code/transformer/output_sample/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/gt_val",file))
		normed_img0 = lpips.im2tensor(img0)
		img1 = lpips.load_image(os.path.join(dir,file))
		normed_img1 = lpips.im2tensor(img1)
		idx += 1

		if(use_gpu):
			normed_img0 = normed_img0.cuda()
			normed_img1 = normed_img1.cuda()

		# Compute distance
		dist01 = loss_fn.forward(normed_img0,normed_img1)
		psnr01 = calculate_psnr(img0, img1)
		ssim01 = calculate_ssim(img0, img1)
		dist += dist01
		psnr += psnr01
		ssim += ssim01
		#print('%s: %.3f %.3f %.3f'%(file, dist01, psnr01, ssim01))
		f.writelines('%s: %.6f %.6f %.6f\n'%(file, dist01, psnr01, ssim01))

		if idx + 1 == max:
			break
	print(f'Average distance: {float(dist/idx)}, Average psnr: {psnr/idx}, Average ssim: {ssim/idx}')
	f.close()
	return dist/idx, psnr/idx, ssim/idx

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('-d','--dir', type=str, default='/mnt/urchin/kzou/code/transformer/output_sample/dmim_pretrain/dmim_pretrain__vit_base__img192__100ep/vit_val_ep300_1024')
	parser.add_argument('-o','--out', type=str, default='./example_dists.txt')
	parser.add_argument('-v','--version', type=str, default='0.1')
	parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

	opt = parser.parse_args()
	eval(opt.dir, opt.out, opt.use_gpu, opt.version)
