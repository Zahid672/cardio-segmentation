import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels, first_stride, no_of_groups=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            #first convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=no_of_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            
            #2nd convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(num_groups=no_of_groups, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            )
        
    def forward(self, x):
        return self.conv(x)


class extended_nnUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ## encoder 
        self.encoder_input = DoubleConv(in_channels=in_channels, out_channels=64, first_stride=1)
        self.encoder_block1 = DoubleConv(in_channels=64, out_channels=128, first_stride=2)
        self.encoder_block2 = DoubleConv(in_channels=128, out_channels=256, first_stride=2)
        self.encoder_block3 = DoubleConv(in_channels=256, out_channels=512, first_stride=2)
        self.encoder_block4 = DoubleConv(in_channels=512, out_channels=512, first_stride=2)
        self.encoder_block5 = DoubleConv(in_channels=512, out_channels=512, first_stride=2)
        
        
        ## decoder
        self.upsample1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decoder_block1 = DoubleConv(in_channels=256+512, out_channels=256, first_stride=1)
        
        self.upsample2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.decoder_block2 = DoubleConv(in_channels=256+512, out_channels=256, first_stride=1)
        
        self.upsample3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder_block3 = DoubleConv(in_channels=128+256, out_channels=128, first_stride=1)
        
        self.upsample4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder_block4 = DoubleConv(in_channels=64+128, out_channels=64, first_stride=1)
        
        self.upsample5 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.decoder_block5 = DoubleConv(in_channels=32+64, out_channels=32, first_stride=1)
        
        
        ## output
        self.deep_supervision3 = nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=1, stride=1)
        self.deep_supervision2 = nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, stride=1)
        self.deep_supervision1 = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, stride=1)
        self.output = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1, stride=1)
        
        
    def forward(self, x):
        ## Encoder
        x0_encoder = self.encoder_input(x)
        x1_encoder = self.encoder_block1(x0_encoder)
        x2_encoder = self.encoder_block2(x1_encoder)
        x3_encoder = self.encoder_block3(x2_encoder)
        x4_encoder = self.encoder_block4(x3_encoder)
        x5_encoder = self.encoder_block5(x4_encoder)
        
        ## Decoder 
        upsample1 = self.upsample1(x5_encoder)
        concat = torch.cat((upsample1, x4_encoder), dim=1)
        x1_decoder = self.decoder_block1(concat)
        
        upsample2 = self.upsample2(x1_decoder)
        concat = torch.cat((upsample2, x3_encoder), dim=1)
        x2_decoder = self.decoder_block2(concat)
        
        upsample3 = self.upsample3(x2_decoder)
        concat = torch.cat((upsample3, x2_encoder), dim=1)
        x3_decoder = self.decoder_block3(concat)
        
        upsample4 = self.upsample4(x3_decoder)
        concat = torch.cat((upsample4, x1_encoder), dim=1)
        x4_decoder = self.decoder_block4(concat)
        
        upsample5 = self.upsample5(x4_decoder)
        concat = torch.cat((upsample5, x0_encoder), dim=1)
        x5_decoder = self.decoder_block5(concat)
        
        aux_output_3 = self.deep_supervision3(x2_decoder)
        aux_output_2 = self.deep_supervision2(x3_decoder)
        aux_output_1 = self.deep_supervision1(x4_decoder)
        output = self.output(x5_decoder)
        
        # return [aux_output_1, aux_output_2, aux_output_3], output
        return output
        
        
if __name__ == "__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  image_size = 128
  x = torch.Tensor(1, 3, image_size, image_size, image_size)
  x.to(device)
  print("x size: {}".format(x.size()))
  
  model = extended_nnUNet(in_channels=3, out_channels=4)

  
  out = model(x)
  print("out size: {}".format(out.size()))    