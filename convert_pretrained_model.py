import paddorch
from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
from glob import glob
import utils
import os

input_weight_folder="best_weigths/BigGAN_C10_seed0_Gch64_Dch64_bs128_nDs4_Glr2.0e-04_Dlr2.0e-04_Gnlrelu_Dnlrelu_GinitN02_DinitN02_ema"

import json

if __name__ == '__main__':
  from paddle import fluid
  place=fluid.CUDAPlace(0)
  with fluid.dygraph.guard(place=place):
        config=json.load( open("c10_config.json", 'r'))

        config['G_activation'] = utils.activation_dict[config['G_nl']]
        config['D_activation'] = utils.activation_dict[config['D_nl']]
        # By default, skip init if resuming training.
        if config['resume']:
            print('Skipping initialization for training resumption...')
            config['skip_init'] = True

        config = utils.update_config_roots(config)
        device = 'cuda'

        # Seed RNG
        utils.seed_rng(config['seed'])

        # Prepare root folders if necessary
        utils.prepare_root(config)

        # Import the model--this line allows us to dynamically select different files.
        model = __import__(config['model'])


        for torch_fn in glob("%s/*pth"%input_weight_folder):
            if "optim" in torch_fn:
                continue # skip optimizer file
            import torch as pytorch
            torch_state_dict= pytorch.load(torch_fn)

            # Next, build the model
            print(torch_fn)
            if os.path.basename(torch_fn).startswith("G"):
                G = model.Generator(**config)
                load_pytorch_pretrain_model(G,torch_state_dict)
                paddorch.save(G.state_dict(),torch_fn.replace(".pth",".pdparams"))
            elif   os.path.basename(torch_fn).startswith("D"):
                D = model.Discriminator(**config)
                load_pytorch_pretrain_model(D,torch_state_dict)
                paddorch.save(D.state_dict(),torch_fn.replace(".pth",".pdparams"))
            else: ##state_dict
                pass #not sure w


