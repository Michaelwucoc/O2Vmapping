import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate
from src.datasets import get_num_semantic_classes

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """
    def __init__(
        self,
        num_input_channels,
        mapping_size = 93,
        scale = 25,
        learnable = True
    ):
        super().__init__()
        
        if learnable : 
            self._B = nn.Parameter(
                torch.randn(
                    (num_input_channels, mapping_size)
                ) * scale
            )
        
        else : 
            self._B = torch.randn(
                (num_input_channels, mapping_size)
            ) * scale
            
    
    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_dim : int,
        out_dim : int,
        activation : str = 'relu',
        *args,
        **kwargs
        ) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)
        
    def reset_parameters(self) -> None :
        torch.nn.init.xavier_uniform_(
            self.weight,
            gain = torch.nn.init.calculate_gain(self.activation)
        )
        if self.bias is not None : 
            torch.nn.init.zeros_(self.bias)




class MLP(nn.Module):
    '''
    Decoder
    Point coordinates used in sampling feature grids / MLP inputs
    
    '''
    def  __init__(
        self,
        name = '', 
        dim = 3,
        c_dim = 32,
        hidden_size = 32,
        n_blocks = 5,
        leaky = False, 
        sample_mode = 'bilinear',
        color = False,
        skips = [2],
        grid_len = 0.16,
        concat_feature = True
        
        ):
        super().__init__()
        
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        
        # classes need + 1 
        # self.num_semantic_classes = get_num_semantic_classes('/home/tiemuer/tiemuer/semantic-SLAM/Datasets/Replica_ICCV/office0/Sequence_1')
        
        if c_dim != 0:
            self.fc_c = nn.ModuleList(
                [nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]
            )
            
            
        embedding_size = 93
        self.embedder = GaussianFourierFeatureTransform(
            dim,
            mapping_size = embedding_size,
            scale = 25
        )
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation = 'relu')] + 
            [DenseLayer(hidden_size, hidden_size, activation = 'relu') 
             if i not in self.skips else 
             DenseLayer(hidden_size + embedding_size, hidden_size, activation = 'relu') for i in range(n_blocks - 1)]
        )
        
        
        if self.color : 
            self.output_linear = DenseLayer(hidden_size, 4, activation = 'linear')
            if self.name == 'semantic':
                self.output_linear = DenseLayer(hidden_size + 16, self.num_semantic_classes + 1, activation = 'linear')
            
        else :
            self.output_linear = DenseLayer(hidden_size, 1, activation = 'linear')
            
            
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)
            
        self.sample_mode = sample_mode
        
    def sample_gird_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(
            c,
            vgrid,
            padding_mode = 'border',
            align_corners = True,
            mode = self.sample_mode
            ).squeeze(-1).squeeze(-1)
        return c
    
    def forward(self, p, c_grid=None):
        
        # get grid feature and position embedding
        if self.c_dim != 0:
            
            name = self.name if self.name != 'semantic' else 'middle'
                
            c = self.sample_gird_feature(
                p,
                c_grid['grid_' + name]
            ).transpose(1,2).squeeze(0)
            
            if self.name == 'fine':
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_gird_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)
            
        p = p.float()
        embedded_pts = self.embedder(p)
        h = embedded_pts
        
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = self.actvn(h)
            
            if self.c_dim != 0:
                re = self.fc_c[i](c)
                h = h + re
                
            if i in self.skips : 
                h = torch.cat([embedded_pts, h], -1)
                
        
        # semantic concat middle feature mapped
        if (self.concat_feature) and (self.name == 'semantic'):
            with torch.no_grad():
                c_semantic = self.sample_gird_feature(
                    p,
                    c_grid['grid_semantic']
                ).transpose(1,2).squeeze(0)
            h = torch.cat([h, c_semantic], dim = 1)
        
        out = self.output_linear(h)
        
        if not self.color :
            out = out.squeeze(-1)

        return out
            

class decoder(nn.Module):
    '''
    
    decoder 
    
    '''
    
    def __init__(
        self,
        dim = 3, 
        c_dim = 32,
        coarse_grid_len = 2.0,
        middle_grid_len = 0.16,
        fine_grid_len = 0.16,
        color_grid_len = 0.16,
        hidden_size = 32
                 ):
        super().__init__()
        
        self.coarse_decoder = MLP(
            name = 'coarse',
            dim = dim,
            c_dim = c_dim, 
            color = False,
            skips = [2],
            n_blocks = 5,
            hidden_size = hidden_size,
            grid_len = coarse_grid_len
        )
        
        self.middle_decoder = MLP(
            name = 'middle',
            dim = dim,
            c_dim = c_dim, 
            color = False,
            skips = [2],
            n_blocks = 5,
            hidden_size = hidden_size,
            grid_len = middle_grid_len
        )
        self.fine_decoder = MLP(
            name = 'fine',
            dim = dim,
            c_dim = c_dim * 2, 
            color = False,
            skips = [2],
            n_blocks = 5,
            hidden_size = hidden_size,
            grid_len = fine_grid_len,
            concat_feature = False
        )
        self.color_decoder = MLP(
            name = 'color',
            dim = dim,
            c_dim = c_dim, 
            color = True,
            skips = [2],
            n_blocks = 5,
            hidden_size = hidden_size,
            grid_len = color_grid_len
        )
        # self.semantic_decoder = MLP(
        #     name = 'semantic',
        #     dim = dim,
        #     c_dim = c_dim, 
        #     color = True,
        #     skips = [2],
        #     n_blocks = 5,
        #     hidden_size = hidden_size,
        #     grid_len = color_grid_len
        # )
        
    def forward(self, p, c_grid, stage='middle', **kwargs):
        
        device = f'cuda:{p.get_device()}'
        
        if stage == 'coarse':
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            raw[..., -1] = occ
            return raw
        elif stage == 'middle' : 
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
            return raw
        elif stage == 'fine':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw
        elif stage == 'color':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = self.color_decoder(p, c_grid)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ + middle_occ
            return raw
            
        # elif stage == 'semantic':
        #     # fine_occ = self.fine_decoder(p, c_grid)
        #     raw = self.semantic_decoder(p, c_grid)
        #     # middle_occ = self.middle_decoder(p, c_grid)
        #     # middle_occ = middle_occ.squeeze(0)
        #     # raw[..., -1] = fine_occ + middle_occ
        #     return raw
        
        return raw