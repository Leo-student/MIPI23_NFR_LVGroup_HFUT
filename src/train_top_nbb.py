from options import TrainOptions
opt = TrainOptions().parse()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx

from log import Log


import time


from trainer_restored import Trainer

import torch 
def main(opt):
    log = Log(__name__, opt).getlog()
    
    trainer = Trainer(opt)
    
    start_epoch = trainer.load_epoch_idx()
    
    start_time = time.time()
    for epoch in range(start_epoch, opt.n_epochs + 1):
        trainer.train(epoch)

        if (epoch) % opt.val_gap == 0:
            trainer.val(epoch)
            trainer.generate()
            trainer.evaluate(epoch)
    end_time = time.time()
    execution_time  =  end_time - start_time 
    log.info("Total training time:  {:.3f} hours and {:.3f} days".format(execution_time/3600, execution_time/3600/24))
            
    


if __name__ == '__main__':
   
    torch.manual_seed(opt.seed)
    # torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.benchmark = True
    
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    main(opt)