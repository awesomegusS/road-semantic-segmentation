import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    
        self.vgg = models.vgg16(pretrained=True).features # selects only conv net
        
        for name, m in self.vgg.named_parameters():  # freeze parameters of pre-trained VGG16 model
            m.requires_grad = False

    def forward(self, x):
        # Get the needed layers' outputs for building FCN-VGG16
        layers = {'16' : 'MaxPool2d_3_out',
                  '23' : 'MaxPool2d_4_out',
                  '30' : 'MaxPool2d_7_out'}

        features = {}

        for index in self.vgg._modules:
            layer = self.vgg._modules[index]
            x = layer(x) # forward pass through model

            # store layer feature activation/map for desired maxpool layers
            if str(index) in layers.keys():
                features[layers[str(index)]] = x


        return features['MaxPool2d_3_out'], features['MaxPool2d_4_out'], features['MaxPool2d_7_out']


class Decoder(nn.Module):
    def __init__(self, num_classes=1):
        super(Decoder, self).__init__()

        ## Decoder parameters
        self.vgg_layer3_depth = 256
        self.vgg_layer4_depth = 512
        self.vgg_layer7_depth = 512
        self.num_classes = num_classes
        self.height = 256
        self.width = 256

         ## Decoder layers definition

        # BUILD the 1x1 CONV SAMPLING AND SKIP CONNECTIONS - Maintain H and W, BRING VOLUME DEPTH IN SYNC WITH DECODER

        self.skip_vgg_layer4 = nn.Conv2d(in_channels = self.vgg_layer4_depth, out_channels = 256,
                                         kernel_size = (1,1), stride = 1, padding = 0)

        self.skip_vgg_layer3 = nn.Conv2d(in_channels = self.vgg_layer3_depth, out_channels = 128,
                                         kernel_size = (1,1), stride = 1, padding = 0)

        # BatchNorm Layers
        #self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(16)

        # Deconv Layers
        self.deconv1 = nn.ConvTranspose2d(in_channels= 512, out_channels= 256,
                                                   kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        self.deconv2 = nn.ConvTranspose2d(in_channels= 256, out_channels= 128,
                                                   kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        self.deconv3 = nn.ConvTranspose2d(in_channels= 128, out_channels= 64,
                                                   kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        self.deconv4 = nn.ConvTranspose2d(in_channels= 64, out_channels= 32,
                                                   kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        self.deconv5 = nn.ConvTranspose2d(in_channels= 32, out_channels= 16,
                                                   kernel_size=2, stride=2, padding=0, dilation=1, output_padding=0)

        self.AMP = nn.AdaptiveMaxPool3d(output_size = (self.num_classes, self.height, self.width))


        # Initialize decoder layers using Xavier's initialization
        self.model_init()


    def forward(self, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out):

        ## Decoder forward useful parameters
        self.batch_size = vgg_layer3_out.shape[0]
        self.vgg_layer3_out = vgg_layer3_out
        self.vgg_layer4_out = vgg_layer4_out
        self.vgg_layer7_out = vgg_layer7_out


        # PASS VGG OUTPUTS THRU 1x1 CONV LAYERS

        self.vgg_layer4_logits = self.skip_vgg_layer4(self.vgg_layer4_out)          # 128 x 14 x 14 (for 224x224)

        self.vgg_layer3_logits = self.skip_vgg_layer3(self.vgg_layer3_out)          # 64 x 28 x 28  (for 224x224)


        # FEED FORWARD DECODER

        # Upsampling H,W by 2
        x = F.relu_(self.deconv1( self.vgg_layer7_out))

        # Skip connection
        x = self.bn1(x.add(self.vgg_layer4_logits))

        # Upsampling H,W by 2
        x = F.relu_(self.deconv2(x))

        # Skip connection
        x = self.bn2(x.add(self.vgg_layer3_logits))

        # Upsampling H,W by 8
        x = self.bn3(F.relu_(self.deconv3(x)))


        # Upsampling H,W by 16
        x = self.bn4(F.relu_(self.deconv4(x)))

        # Upsampling H,W by 32
        x = self.bn5(F.relu_(self.deconv5(x)))

        # Bring feature depth to num_classes
        output = self.AMP(x)

        # We ensure appropriate Tensor shape:  batchsize x num_classes x H x W
        output = output.view(self.batch_size,self.num_classes, self.height, self.width)

        return output

    def model_init(self):
        # We initialize the decoder parameters using Xavier's approach
        torch.nn.init.xavier_uniform_(self.deconv1.weight)
        torch.nn.init.xavier_uniform_(self.deconv2.weight)
        torch.nn.init.xavier_uniform_(self.deconv3.weight)
        torch.nn.init.xavier_uniform_(self.deconv4.weight)
        torch.nn.init.xavier_uniform_(self.deconv5.weight)



def enforce_orient(image):
        """ Helper function to enforce the orientation of original picture
            Returns: Np array of image
        """
        
        if hasattr(image, '_getexif'):  # Check if image has EXIF data
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(0x0112)
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        return np.array(image) # return an np array of image