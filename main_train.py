from config import get_arguments
from SinGAN.manipulate import *
from SinGAN.training import *
import SinGAN.functions as functions
from SinGAN.imresize import torch2uint8
from vglc.vglc_utils import save_sc2_image
from skimage import io as img

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--mode', help='task to be done', default='train')

    #  NOAM VGLC
    parser.add_argument('--vglc_json', help='(VGLC) Enable and set game data JSON', default=None)

    opt = parser.parse_args()
    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
        real = functions.read_image(opt)
        if opt.vglc_json or True:
            im = torch2uint8(real, opt)
            if opt.vglc_json:
                save_sc2_image(im, dir2save + "/input.png")
            else:
                img.imsave(dir2save + "/input.png", im)
        functions.adjust_scales2image(real, opt)
        train(opt, Gs, Zs, reals, NoiseAmp)
        SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt)
