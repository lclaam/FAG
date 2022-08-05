import os
import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from gan_module import Generator
from collections import OrderedDict

parser = ArgumentParser()
parser.add_argument('--image_dir', default='images', help='The image directory')
parser.add_argument('--save_path', default='output', help='The result directory')
parser.add_argument('--save_name', default='mygraph.png', help='The result filename')
parser.add_argument('--model_path', default='pretrained_model/epoch=4-step=11889.ckpt', help='The model file')

@torch.no_grad()
def main():
    args = parser.parse_args()
    image_paths = [os.path.join(args.image_dir, x) for x in os.listdir(args.image_dir) if
                   x.endswith('.png') or x.endswith('.jpg')]
    save_to = os.path.join(args.save_path, args.save_name)

    model = Generator(ngf=32, n_residual_blocks=9)
    
    # use the pretrained model (generator model extracted)
    # ckpt = torch.load('pretrained_model/state_dict.pth', map_location='cpu')
    
    # use the GAN model you train (genrator model and discriminator model not separated)
    ckpt = torch.load(args.model_path, map_location='cpu')['state_dict']
    ckptA2B = OrderedDict()     # generator A2B (young to old)
    ckptB2A = OrderedDict()     # generator B2A (old to young?)
    for key, value in ckpt.items():
        if 'genA2B' in key:
            ckptA2B[key.split('.',1)[1]] = value
        elif 'genB2A' in key:
            ckptB2A[key.split('.',1)[1]] = value
        
    model.load_state_dict(ckptA2B)
    model.eval()
    trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    nr_images = len(image_paths) if len(image_paths) < 6 else 6
    fig, ax = plt.subplots(2, nr_images, figsize=(20, 10))
    random.shuffle(image_paths)
    for i in range(nr_images):
        img = Image.open(image_paths[i]).convert('RGB')
        img = trans(img).unsqueeze(0)
        aged_face = model(img)
        aged_face = (aged_face.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0
        ax[0, i].imshow((img.squeeze().permute(1, 2, 0).numpy() + 1.0) / 2.0)
        ax[1, i].imshow(aged_face)
    # plt.show()
    plt.savefig(save_to)

if __name__ == '__main__':
    main()
