from options import TrainOptions
opt = TrainOptions().parse()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx

from log import Log
log = Log(__name__).getlog()

from trainer import Trainer

import torch 
def main(opt):
    start_epoch = 1
    trainer = Trainer(opt)
    for epoch in range(start_epoch, opt.n_epochs + 1):
        trainer.train(epoch)

        if (epoch) % opt.val_gap == 0:
            trainer.val(epoch)
            trainer.generate()
            trainer.evaluate(epoch)
            
    


if __name__ == '__main__':
    
    torch.manual_seed(opt.seed)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    main(opt)