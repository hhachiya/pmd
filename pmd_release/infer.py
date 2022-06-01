import torch
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import os
import time
from argparse import ArgumentParser
from PIL import Image
import skimage.io
import pdb

from config import msd_testing_root
from misc import check_mkdir, crf_refine

from model.pmd import PMD
device_ids = [0]
torch.cuda.set_device(device_ids[0])



def parse_args():
    parser = ArgumentParser(description="pmd trainer")
    parser.add_argument('-data',type=str,default='split',help='name of dataset, e.g., split and theta, default: split')
    parser.add_argument('-is_crf','--is_crf',action='store_true')
    parser.add_argument('-scale','--scale',type=int,default='384')

    return parser.parse_args()

args = parse_args()
'''
args = {
    'scale': 384,
    'crf': True
}
'''

img_transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if args.data == 'split':
    test_dir = '../data/PMD_split/PMD/test/'
    to_test = {'ADE20K': f'{test_dir}ADE20K', 'COCO': f'{test_dir}COCO', 'MINC': f'{test_dir}MINC', 'NYUD': f'{test_dir}NYUD', 'PASCAL': f'{test_dir}PASCAL', 'SUNRGBD': f'{test_dir}SUNRGBD'}
    model_path = 'model/pmd_split.pth'

elif args.data == 'theta':
    test_dir = '../data/theta/test/'
    to_test = {'theta': f'{test_dir}'}
    model_path = 'model/pmd_theta.pth'

to_pil = transforms.ToPILImage()


def main():
    net = PMD().cuda(device_ids[0])
    net.load_state_dict(torch.load(model_path)["model_state_dict"])
    
    net.eval()
    with torch.no_grad():
        for name, root in to_test.items():
            img_list = [img_name for img_name in os.listdir(os.path.join(root, 'image'))]
            start = time.time()
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(name, idx + 1, len(img_list)))
                check_mkdir(os.path.join("result", name))

                img = Image.open(os.path.join(root, 'image', img_name))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    print("{} is a gray image.".format(name))
                w, h = img.size
                img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
                # f_4, f_3, f_2, f_1, e = net(img_var)
                f_4, f_3, f_2, f_1, edge, final = net(img_var)
                # output = f.data.squeeze(0).cpu()
                # edge = e.data.squeeze(0).cpu()
                f_4 = f_4.data.squeeze(0).cpu()
                f_3 = f_3.data.squeeze(0).cpu()
                f_2 = f_2.data.squeeze(0).cpu()
                f_1 = f_1.data.squeeze(0).cpu()
                edge = edge.data.squeeze(0).cpu()
                final = final.data.squeeze(0).cpu()

                f_4 = np.array(transforms.Resize((h, w))(to_pil(f_4)))
                f_3 = np.array(transforms.Resize((h, w))(to_pil(f_3)))
                f_2 = np.array(transforms.Resize((h, w))(to_pil(f_2)))
                f_1 = np.array(transforms.Resize((h, w))(to_pil(f_1)))
                edge = np.array(transforms.Resize((h, w))(to_pil(edge)))
                final = np.array(transforms.Resize((h, w))(to_pil(final)))

                #if args['crf']:
                if args.is_crf:
                    final = crf_refine(np.array(img.convert('RGB')), final)

                #Image.fromarray(final).save(os.path.join("result", name, img_name[:-4] + ".png"))
                final = np.expand_dims(final,-1)
                final = Image.fromarray(np.concatenate([final,np.zeros_like(final),np.zeros_like(final)],axis=-1)) 
                Image.blend(final, img, alpha=0.4).save(os.path.join("result", name, img_name[:-4] + ".png"))

            end = time.time()
            print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()
