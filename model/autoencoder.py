
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super(AutoEncoder, self).__init__()
        
#         # Definisci l'encoder
#         # Cambia i parametri di input e output dei layer
#         self.encoder_16 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
        
#         # Decoder
#         self.decoder_16 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
#             nn.Sigmoid()
#         )

#         # Encoder
#         self.encoder_8 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
        
#         # Decoder
#         self.decoder_8 = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         print(x.shape)
#         if x.shape[0] == 8:
#             encoded = self.encoder_8(x)
#             decoded = self.decoder_8(encoded)
#             return decoded
#         else:
#             encoded = self.encoder_16(x)
#             decoded = self.decoder_16(encoded)
#             return decoded

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Definisci l'encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Definisci il decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Crea un'istanza del tuo autoencoder
autoencoder = Autoencoder()
