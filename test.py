#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.segformer import SegFormer

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 150
    phi             = 'b5'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model   = SegFormer(num_classes = num_classes, phi = phi, pretrained=False).to(device)
    
    state = torch.load("model_data/segformer.b5.640x640.ade.160k.pth")['state_dict']
    model.load_state_dict(state, strict=False)
    torch.save(model.backbone.state_dict(), "model_data/segformer_" + phi + "_backbone_weights.pth")
    