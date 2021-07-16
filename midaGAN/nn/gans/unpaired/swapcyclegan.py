import itertools
from dataclasses import dataclass

import torch
from midaGAN import configs
from midaGAN.data.utils.image_pool import ImagePool
from midaGAN.nn.gans.base import BaseGAN
from midaGAN.nn.losses.adversarial_loss import AdversarialLoss
from midaGAN.nn.losses.swapcyclegan_losses import SwapCycleGANLosses


@dataclass
class OptimizerConfig(configs.base.BaseOptimizerConfig):
    """SwapGAN Optimizer Config"""
    lambda_AB: float = 10.0
    lambda_BA: float = 10.0
    lambda_identity: float = 10.0
    proportion_ssim: float = 0


@dataclass
class SwapCycleGANConfig(configs.base.BaseGANConfig):
    """SwapCycleGAN Config"""
    name: str = "SwapCycleGAN"
    pool_size: int = 50
    optimizer: OptimizerConfig = OptimizerConfig


class SwapCycleGAN(BaseGAN):
    """SwapCycleGAN architecture, modified from the paper:
    `Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks`, Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros, ICCV, 2017`
    We separate the downsampling and upsampling part of the generator networks and swap them when calculating the identity loss. 
    This way, the downsampler for domain A is alway used for input from domain A and the upsampler toward domain B is always used for output to domain B.
    """

    def __init__(self, conf):
        super().__init__(conf)

        # Inputs and Outputs of the model
        visual_names = ['real_A', 'fake_B', 'rec_A', 'idt_A', 'real_B', 'fake_A', 'rec_B', 'idt_B']
        # initialize the visuals as None
        self.visuals = {name: None for name in visual_names}

        # Losses used by the model
        loss_names = ['cycle_A', 'cycle_B', 'idt_A', 'idt_B', 'discr_A', 'discr_B']
        self.losses = {name: None for name in loss_names}

        # Optimizers
        optimizer_names = ['CG', 'D']
        self.optimizers = {name: None for name in optimizer_names}

        # Compressors, Generators and Discriminators
        network_names = ['C_A', 'C_B', 'G_A', 'G_B', 'D_A', 'D_B'] if self.is_train else ['C_A', 'G_B']
        self.networks = {name: None for name in network_names}

        if self.is_train:
            # Create image buffer to store previously generated images
            self.fake_A_pool = ImagePool(conf.train.gan.pool_size)
            self.fake_B_pool = ImagePool(conf.train.gan.pool_size)

        # Set up networks, optimizers, schedulers, mixed precision, checkpoint loading, network parallelization...
        self.setup()

    def init_criterions(self):
        # Standard GAN loss
        self.criterion_adv = AdversarialLoss(
            self.conf.train.gan.optimizer.adversarial_loss_type).to(self.device)
        # Generator-related losses -- Cycle-consistency and Identity loss
        self.criterion_G = SwapCycleGANLosses(self.conf)

    def init_optimizers(self):
        lr_G = self.conf.train.gan.optimizer.lr_G
        lr_D = self.conf.train.gan.optimizer.lr_D
        beta1 = self.conf.train.gan.optimizer.beta1
        beta2 = self.conf.train.gan.optimizer.beta2

        params_CG = itertools.chain(self.networks['C_A'].parameters(), self.networks['C_B'].parameters(), self.networks['G_A'].parameters(), self.networks['G_B'].parameters())
        params_D = itertools.chain(self.networks['D_A'].parameters(), self.networks['D_B'].parameters())

        self.optimizers['CG'] = torch.optim.Adam(params_CG, lr=lr_G, betas=(beta1, beta2))
        self.optimizers['D'] = torch.optim.Adam(params_D, lr=lr_D, betas=(beta1, beta2))

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
        compressors = [self.networks['D_A'], self.networks['D_B']]
        generators = [self.networks['G_A'], self.networks['G_B']]
        discriminators = [self.networks['D_A'], self.networks['D_B']]

        self.forward()  # compute fake images and reconstruction images.

        # Compute generator based metrics dependent on visuals
        self.metrics.update(self.training_metrics.compute_metrics_G(self.visuals))

        # ------------------------ C,G_A/B ----------------------------------------------------------
        self.set_requires_grad(compressors, True)
        self.set_requires_grad(generators, True)
        self.set_requires_grad(discriminators, False)

        self.optimizers['CG'].zero_grad(set_to_none=True)
        self.backward_G()  # calculate gradients for CG
        self.optimizers['CG'].step()  # update weights in CG
        # ------------------------ D_A/B ----------------------------------------------------------
        self.set_requires_grad(compressors, False)
        self.set_requires_grad(generators, False)
        self.set_requires_grad(discriminators, True)
        self.optimizers['D'].zero_grad(set_to_none=True)
        self.backward_D('D_B')  # calculate gradients for D_B

        # Update metrics for D_B
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_B', self.pred_real, self.pred_fake))

        self.backward_D('D_A')  # calculate gradients for D_A

        # Update metrics for D_A
        self.metrics.update(
            self.training_metrics.compute_metrics_D('D_A', self.pred_real, self.pred_fake))

        self.optimizers['D'].step()  # update D_B and D_A's weights
        # -----------------------------------------------------------------------------------------

    def forward(self):
        """Run forward pass; called by both methods <optimize_parameters> and <test>."""
        real_A = self.visuals['real_A']
        real_B = self.visuals['real_B']

        # Apply the compressors
        compr_A = self.networks['C_A'](real_A)
        compr_B = self.networks['C_B'](real_B)

        # Apply the generators
        idt_A = self.networks['G_A'](compr_A)
        fake_A = self.networks['G_A'](compr_B)
        idt_B = self.networks['G_B'](compr_B)
        fake_B = self.networks['G_B'](compr_A)

        # Apply the compressors again for the full cycle
        compr_A = self.networks['C_A'](fake_A)
        compr_B = self.networks['C_B'](fake_B)

        # Apply the generators again to complete the cycle 
        rec_A = self.networks['G_A'](compr_B)
        rec_B = self.networks['G_B'](compr_A)

        self.visuals.update({
            'fake_B': fake_B,
            'rec_A': rec_A,
            'idt_A': idt_A,
            'fake_A': fake_A,
            'rec_B': rec_B,
            'idt_B': idt_B
        })

    def backward_D(self, discriminator):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        Also calls backward() on loss_D to calculate the gradients.
        """
        if discriminator == 'D_B':
            real = self.visuals['real_B']
            fake = self.visuals['fake_B']
            fake = self.fake_B_pool.query(fake)
        elif discriminator == 'D_A':
            real = self.visuals['real_A']
            fake = self.visuals['fake_A']
            fake = self.fake_A_pool.query(fake)
        else:
            raise ValueError('The discriminator has to be either "D_A" or "D_B".')

        self.pred_real = self.networks[discriminator](real)

        # Detaching fake: https://github.com/pytorch/examples/issues/116
        self.pred_fake = self.networks[discriminator](fake.detach())

        loss_real = self.criterion_adv(self.pred_real, target_is_real=True)
        loss_fake = self.criterion_adv(self.pred_fake, target_is_real=False)
        self.losses[discriminator] = loss_real + loss_fake

        # backprop
        self.backward(loss=self.losses[discriminator], optimizer=self.optimizers['D'], loss_id=2)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B using all specified losses"""

        fake_A = self.visuals['fake_A']  # G_A(C_B(B))
        fake_B = self.visuals['fake_B']  # G_B(C_A(A))

        # ------------------------- Discriminator loss ----------------------------
        pred_A = self.networks['D_A'](fake_A)  # D_A(G_A(C_B(B)))
        pred_B = self.networks['D_B'](fake_B)  # D_B(G_B(C_A(A)))

        self.losses['discr_A'] = self.criterion_adv(pred_B, target_is_real=True)
        self.losses['discr_B'] = self.criterion_adv(pred_A, target_is_real=True)
        # ---------------------------------------------------------------

        # ------------- Cycle, Identity loss -------------
        losses = self.criterion_G(self.visuals)
        self.losses['cycle_A'] = losses['cycle_A']
        self.losses['cycle_B'] = losses['cycle_B']
        self.losses['idt_A'] = losses['idt_A']
        self.losses['idt_B'] = losses['idt_B']
        # ---------------------------------------------------------------

        # combine losses and calculate gradients
        combined_loss_G = self.losses['cycle_A'] + self.losses['cycle_B'] + self.losses['idt_A'] + self.losses['idt_B'] + self.losses['discr_A'] + self.losses['discr_B']
        self.backward(loss=combined_loss_G, optimizer=self.optimizers['CG'], loss_id=0, retain_graph=True)

    def infer(self, input, inputDomain='A', outputDomain='B'):
        assert inputDomain in ['A', 'B'], "Specify the domain for the input, A or B."
        assert outputDomain in ['A', 'B'], "Specify the domain for the output, A or B."

        assert f'C_{inputDomain}' in self.networks.keys()
        assert f'G_{outputDomain}' in self.networks.keys()

        with torch.no_grad():
            compressed = self.networks[f'C_{inputDomain}'](input)
            return self.networks[f'G_{outputDomain}'](compressed)
