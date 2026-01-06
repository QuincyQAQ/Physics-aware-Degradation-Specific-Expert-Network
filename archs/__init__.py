import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info

from .DeMoE import DeMoE       


def create_model(opt, local_rank, global_rank=1):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    name = opt['name']
        
    if name == 'DeMoE':
        model = DeMoE(img_channel=opt['img_channels'],
                width=opt['width'], 
                middle_blk_num=opt['middle_blk_num'], 
                enc_blk_nums=opt['enc_blk_nums'],
                dec_blk_nums=opt['dec_blk_nums'],
                num_exp=opt['num_experts'],
                k_used=opt['k_used'])

    else:
        raise NotImplementedError('This network is not implemented')
    if global_rank ==0:
        print( '**************************** \n',f'Using {name} network')

        input_size = (3, 256, 256)
        macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
        print(f' ---- Computational complexity at {input_size}: {macs}')
        print(' ---- Number of parameters: ', params)    
    else:
        macs, params = None, None

    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{local_rank}') if use_cuda else torch.device('cpu')
    model.to(device)
    
    if 'find_unused_params' in opt: find_unused_params = opt['find_unused_params']
    else: find_unused_params = False
 
    use_ddp = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    if use_ddp:
        ddp_device_ids = [local_rank] if use_cuda else None
        model = DDP(model, device_ids=ddp_device_ids, find_unused_parameters=find_unused_params)
    
    return model, macs, params

def load_weights(model, model_weights, global_rank = 1):
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in model_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)

    total_checkpoint_keys = len(model_weights)
    total_model_keys = len(new_weights)
    matching_keys = len(set(model_weights.keys()) & set(new_weights.keys()))

    if global_rank==0:
        print(f"Total keys in checkpoint: {total_checkpoint_keys}")
        print(f"Total keys in model state dict: {total_model_keys}")
        print(f"Number of matching keys: {matching_keys}")

    return model

def resume_model(model,
                 path_model, 
                 local_rank,
                 global_rank,resume:str=None):
    
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    if torch.cuda.is_available():
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    else:
        map_location = 'cpu'
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints['params']
        model = load_weights(model, model_weights=weights,global_rank = global_rank)
        if global_rank == 0: print(' ---- Loaded weights', '\n ***************************')
    else:
        if global_rank==0: print(' ---- Starting from zero the training', '\n ***************************')
    
    return model


__all__ = ['create_model', 'resume_model', 'load_weights']



    
