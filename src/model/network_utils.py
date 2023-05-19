from torch import nn
from functools import partial
from torch.nn import init


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        
        print("Network [{}] was created. Total number of parameters: {:.1f} M. ".format(self.__class__.__name__, num_params / 1e6))
           
    
    def init_weights(self ,init_type, mean = 0 , std = 0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, std)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
                elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                    if init_type == 'normal':
                        init.normal_(m.weight.data, 0.0, std)
                    elif init_type == 'xavier':
                        init.xavier_normal_(m.weight.data, std=std)
                    elif init_type == 'xavier_uniform':
                        init.xavier_uniform_(m.weight.data, std=1.0)
                    elif init_type == 'kaiming':
                        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                    elif init_type == 'orthogonal':
                        init.orthogonal_(m.weight.data, std=std)
                    elif init_type == 'none':  # uses pytorch's default init method
                        m.reset_parameters()
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                    if hasattr(m, 'bias') and m.bias is not None:
                        init.constant_(m.bias.data, 0.0)
            # init.normal_(m.weight.data , mean , std)
            # init.constant_(m.bias.data , 0)
        self.apply(init_func)
        
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, std)
        
    
    def forward(self , *inputs):
        pass




#------------------------------
# Sets the padding method for the input
def get_pad_layer(type):
    # Chooses reflection , places mirror around boundary and reflects the value
    if type == "reflection":
        layer = nn.ReflectionPad2d
    # Replicates the padded area with nearest boundary value
    elif type == "replication":
        layer = nn.ReplicationPad2d
    # Padding of Image with constat 0 
    elif type == "zero":
        layer = nn.ZeroPad2d
    else:
        raise NotImplementedError("Padding type {} is not valid . Please choose among ['reflection' ,'replication' ,'zero']".format(type))
    
    return layer

    
#----------------------------------
# Sets the norm layer 
def get_norm_layer(type):
    if type == "BatchNorm2d":
        layer = partial(nn.BatchNorm2d , affine = True) 
    elif type == "InstanceNorm2d":
        layer = partial(nn.InstanceNorm2d ,affine = False)
    else : 
        raise NotImplementedError("Norm type {} is not valid. Please choose ['BatchNorm2d' , 'InstanceNorm2d']".format(type))
    
    return layer

    
