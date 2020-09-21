import torch
import torch.nn.functional as F

import itertools
from apex import amp

from midaGAN.data.utils.image_pool import ImagePool
from midaGAN.nn.generators import build_G
from midaGAN.nn.discriminators import build_D
from midaGAN.nn.gans.basegan import BaseGAN

from midaGAN.nn.losses.generator_loss import GeneratorLoss
from midaGAN.nn.losses.gan_loss import GANLoss

# Config imports
from dataclasses import dataclass, field
from omegaconf import MISSING
from midaGAN.conf import BaseGANConfig


@dataclass
class CycleGANConfig(BaseGANConfig):
    """CycleGAN"""
    name: str = "CycleGAN"


class CycleGAN(BaseGAN):
    """CycleGAN"""

    def __init__(self, conf):
        super().__init__(conf)
        
        # Inputs and Outputs of the model
        visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_B', 'real_B', 'fake_A', 'rec_B', 'idt_A']
        self.visuals = {name: None for name in visual_names}

        # Losses used by the model
        loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'inv_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'inv_B']
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['G', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Generators and Discriminators
        network_names = ['G_A', 'G_B', 'D_A', 'D_B'] if self.is_train else ['G_A'] # during test time, only G
        self.networks = {name: None for name in network_names}

        # Initialize Generators and Discriminators
        self.init_networks(conf)

        # TODO: move to basegan
        if self.is_train:
            # Intialize loss functions (criterions) and optimizers
            self.init_criterions(conf)
            self.init_optimizers(conf)

            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(conf.gan.pool_size)
            self.fake_B_pool = ImagePool(conf.gan.pool_size)

        self.setup() # schedulers, mixed precision, checkpoint loading and network parallelization


    def init_networks(self, conf):
        # TODO: move it to basegan 
        for name in self.networks.keys():
            if name.startswith('G'):
                self.networks[name] = build_G(conf, self.device)
            elif name.startswith('D'):
                self.networks[name] = build_D(conf, self.device)
            else:
                raise ValueError('Network\'s name has to begin with either "G" if it is a generator, \
                                  or "D" if it is a discriminator.')


    def init_criterions(self, conf):
        # TODO: move it to basegan
        # Standard GAN loss 
        self.criterion_gan = GANLoss(conf.gan.loss_type).to(self.device)
        # Generator-related losses -- Cycle-consistency, Identity and Inverse loss
        self.criterion_G = GeneratorLoss(conf)


    def init_optimizers(self, conf):
        lr_G = conf.optimizer.lr_G
        lr_D = conf.optimizer.lr_D
        beta1 = conf.optimizer.beta1

        params_G = itertools.chain(self.networks['G_A'].parameters(), 
                                   self.networks['G_B'].parameters()) 
        params_D = itertools.chain(self.networks['D_A'].parameters(), 
                                   self.networks['D_B'].parameters())         

        self.optimizers['G'] = torch.optim.Adam(params_G, lr=lr_G, betas=(beta1, 0.999)) 
        self.optimizers['D'] = torch.optim.Adam(params_D, lr=lr_D, betas=(beta1, 0.999))                            


    def set_input(self, input):
        """Unpack input data from the dataloader.
        Parameters:
            input (dict) -- a pair of data samples from domain A and domain B.
        """
        self.visuals['real_A'] = input['A'].to(self.device)
        self.visuals['real_B'] = input['B'].to(self.device)


    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights. 
        Called in every training iteration.
        """
        self.forward()  # compute fake images and reconstruction images.

        discriminators = [ self.networks['D_A'], self.networks['D_B'] ]
        # ------------------------ G (A and B) ----------------------------------------------------
        self.set_requires_grad(discriminators, False)   # Ds require no gradients when optimizing Gs
        self.optimizers['G'].zero_grad()                # set G's gradients to zero
        self.backward_G(loss_id=0)                      # calculate gradients for G
        self.optimizers['G'].step()                     # update G's weights
        # ------------------------ D_A and D_B ----------------------------------------------------
        self.set_requires_grad(discriminators, True)
        self.optimizers['D'].zero_grad()                #set D_A and D_B's gradients to zero
        self.backward_D('D_A', loss_id=1)               # calculate gradients for D_A
        self.backward_D('D_B', loss_id=2)               # calculate graidents for D_B
        self.optimizers['D'].step()                     # update D_A and D_B's weights
        # -----------------------------------------------------------------------------------------

    
    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Forward cycle G_A (A to B)
        fake_B = self.networks['G_A'](real_A) 
        rec_A  = self.networks['G_B'](fake_B)

        # Backward cycle G_B (B to A)
        fake_A = self.networks['G_B'](real_B) # G forward is G_A, G inverse forward is G_B
        rec_B  = self.networks['G_A'](fake_A)

        self.visuals.update({'fake_B': fake_B, 'rec_A': rec_A, 
                             'fake_A': fake_A, 'rec_B': rec_B})

    
    def backward_D(self, discriminator, loss_id=0):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        if discriminator == 'D_A':
            real = self.visuals['real_B']
            fake = self.visuals['fake_B']
            fake = self.fake_B_pool.query(fake)
            
        elif discriminator == 'D_B':
            real = self.visuals['real_A']
            fake = self.visuals['fake_A']
            fake = self.fake_A_pool.query(fake)
        else:
            raise ValueError('The discriminator has to be either "D_A" or "D_B".')

        pred_real = self.networks[discriminator](real)
        pred_fake = self.networks[discriminator](fake.detach())

        loss_real = self.criterion_gan(pred_real, target_is_real=True)
        loss_fake = self.criterion_gan(pred_fake, target_is_real=False)
        self.losses[discriminator] = loss_real + loss_fake

        # backprop
        self.backward(loss=self.losses[discriminator], optimizer=self.optimizers['D'], loss_id=loss_id)


    def backward_G(self, loss_id=0):
        """Calculate the loss for generators G_A and G_B using all specified losses"""        

        real_A = self.visuals['real_A']        
        real_B = self.visuals['real_B']       
        fake_A = self.visuals['fake_A']  # G_B(B)
        fake_B = self.visuals['fake_B']  # G_A(A)

        # ------------------------- GAN Loss ----------------------------
        pred_A = self.networks['D_A'](fake_B)  # D_A(G_A(A))
        pred_B = self.networks['D_B'](fake_A)  # D_B(G_B(B))
        self.losses['G_A'] = self.criterion_gan(pred_A, target_is_real=True) # Forward GAN loss D_A(G_A(A))
        self.losses['G_B'] = self.criterion_gan(pred_B, target_is_real=True) # Backward GAN loss D_B(G_B(B))
        # ---------------------------------------------------------------

        # ------------- G Losses (Cycle, Identity, Inverse) -------------
        if self.criterion_G.is_using_identity():
            self.visuals['idt_A'] = self.networks['G_A'](real_B)
            self.visuals['idt_B'] = self.networks['G_B'](real_A)
        losses_G = self.criterion_G(self.visuals)
        self.losses.update(losses_G)
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = sum(losses_G.values()) + self.losses['G_A'] + self.losses['G_B']
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['G'], loss_id=loss_id)
