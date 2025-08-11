import torch
import torchvision
import pdb
from termcolor import colored

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)
    #TODO: figure out how to load imagnet or clip resnet weights
    print(colored(f'USING OTHER WEIGHTS: {name}','magenta'))
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

def get_clip(name="ViT-B/32", **kwargs):
    """
    Get CLIP visual encoder
    name: CLIP model name like "ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", etc.
    """
    import clip
    print(colored(f'Loading CLIP model: {name}', 'magenta'))
    
    # Load CLIP model
    model, preprocess = clip.load(name, device='cpu')
    
    # Extract just the visual encoder
    visual_encoder = model.visual
    
    # Make it compatible with the rest of the codebase
    # CLIP's visual encoder outputs a different shape, so we need to handle that
    class CLIPVisualWrapper(torch.nn.Module):
        def __init__(self, visual_encoder):
            super().__init__()
            self.visual = visual_encoder
            # Get the output dimension
            if hasattr(visual_encoder, 'output_dim'):
                self.out_features = visual_encoder.output_dim
            else:
                # For ViT models
                if hasattr(visual_encoder, 'ln_post'):
                    self.out_features = visual_encoder.ln_post.normalized_shape[0]
                # For ResNet models  
                else:
                    self.out_features = visual_encoder.attnpool.c_proj.out_features
                    
        def forward(self, x):
            return self.visual(x)
    
    return CLIPVisualWrapper(visual_encoder)
