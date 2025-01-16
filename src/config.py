import yaml
from src import models



def load_config(path, default_path = None):
    
    '''
        Loads config file.
    '''
    
    with open(path, 'r') as f : 
        cfg_special = yaml.full_load(f)
        
    inherit_from = cfg_special.get('inherit_from')
    
    
    if inherit_from is not None:
        cfg = load_config(inherit_from)
    else :
        cfg = dict()
        
    update_recursive(cfg, cfg_special)
    
    return cfg 

def update_recursive(dict1, dict2):
    
    '''
    update two configs

    Args:
        dict1 (_type_): _description_
        dict2 (_type_): _description_
    '''
    
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else : 
            dict1[k] = v
            
def get_model(cfg):
    
    '''
        return the network model.
    
        Args : cfg
        Returns : decoder

    '''
    
    dim  = cfg['data']['dim']
    
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    
    color_grid_len = cfg['grid_len']['color']
    
    c_dim = cfg['model']['c_dim']
    pos_embedding_method = cfg['model']['pos_embedding_method']
    
    decoder = models.decoder(
        dim = dim, 
        c_dim = c_dim, 
        coarse_grid_len = coarse_grid_len,
        middle_grid_len = middle_grid_len,
        fine_grid_len = fine_grid_len,
        color_grid_len = color_grid_len,
        
    )
    
    return decoder