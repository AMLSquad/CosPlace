import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder_weights_grads = None
        self.encoder_bias_grads = None

            # Definisci il decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            )  
        self.decoder_weights_grads = None
        self.decoder_bias_grads = None
        
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.requires_grad = True
         
          
    def take_grad(self):
        encoder_weigths_grads = []
        encoder_bias_grads =  []
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                encoder_weigths_grads.append(module.weight.grad.detach())
                module.weight.zero_()
                if module.bias is not None:
                    encoder_bias_grads.append(module.bias.grad.detach())
                    module.bias.zero_()

        decoder__weigths_grads = []
        decoder_bias_grads =  []
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                decoder__weigths_grads.append(module.weight.grad.detach())
                module.weight.zero_()

                if module.bias is not None:
                    decoder_bias_grads.append(module.bias.grad.detach())
                    module.bias.zero_()

        self.encoder_weights_grads = encoder_weigths_grads
        self.encoder_bias_grads = encoder_bias_grads
        self.decoder_weights_grads = decoder__weigths_grads
        self.decoder_bias_grads = decoder_bias_grads

    def add_grad(self):
        w = 0
        b = 0
        for module in self.encoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.grad = self.encoder_weights_grads[w]
                w = w + 1
                if module.bias is not None:
                    module.bias.grad = self.encoder_bias_grads[b]
                    b = b + 1

        w = 0
        b = 0
        for module in self.decoder.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                module.weight.grad = self.decoder_weights_grads[w]
                w = w + 1
                if module.bias is not None:
                    module.bias.grad = self.decoder_bias_grads[b]
                    

       
    
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

