import os
import glob
import random
from datetime import datetime
from sqlite3 import ProgrammingError
from venv import create

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"]=""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        hidden_dims = config['enc_hidden_dims'].copy()
        latent_dim = config['latent_dim']
        
        if self.config['model_type'] == 'cvae':
            in_dim = hidden_dims[0] + config['latent_dim']
        else:
            in_dim = hidden_dims[0]

        self.fc1 = nn.Linear(in_dim, hidden_dims[-1])
        self.fc2 = nn.Linear(hidden_dims[-1], latent_dim*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.enc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.enc_logvar = nn.Linear(latent_dim*2, latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, x, se=None):
        if se is not None: x = torch.cat((x, se), dim=-1)
        x = self.lrelu(self.fc1(x))
        x = self.lrelu(self.fc2(x))
        mu = self.enc_mu(x)
        logvar = self.enc_logvar(x)

        # if self.config['encoded_noise'] == True:
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()

        hidden_dims = config['dec_hidden_dims'].copy()
        latent_dim = config['latent_dim']

        if config['model_type'] == 'cvae':
            in_dim = latent_dim * 2
        else:
            in_dim = latent_dim

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        # self.apply(weights_init)
    
    def forward(self, z, se=None):
        if se is not None:
            z = torch.cat((z, se), dim=-1)
            
        x = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc2(x))
        return x
        

class Discriminator_D1(nn.Module):
    def __init__(self, config):
        super(Discriminator_D1, self).__init__()
        
        # if config['model_type'] == 'cvae':
        #     in_dim = config['visual_dim'] + config['attr_dim']
        # else:
        #     in_dim = config['visual_dim']
        in_dim = config['visual_dim']

        self.fc1 = nn.Linear(in_dim, config['d_hdim'])
        self.fc2 = nn.Linear(config['d_hdim'], 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init)
    def forward(self, x, att=None):
        if att is None:
            h = x
        else:
            h = torch.cat((x, att), dim=1)
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        h = self.sigmoid(h)
        return h



# setting for weight init function
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# get dataset
def get_dataset():
    pass



class TrainerGAN():
    def __init__(self, config):
        self.config = config

        self.E = Encoder(self.config)
        self.G = Generator(self.config)
        self.D = Discriminator_D1(self.config)

        self.loss = nn.BCELoss()
        self.mse = nn.MSELoss(reduction='mean')

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        self.opt_E = torch.optim.Adam(self.E.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))

        self.dataloader = None
        self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
        self.ckpt_dir = os.path.join(self.config["workspace_dir"], 'checkpoints')

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO, 
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

        self.steps = 0
        self.z_samples = Variable(torch.randn(100, self.config["latent_dim"])).to(device)

    def prepare_environment(self):
        """
        Use this function to prepare function
        """
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_dir, self.config['dataset'], time+f'_{self.config["model_type"]}')
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.config['dataset'], time+f'_{self.config["model_type"]}')
        os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        # create dataset by the above function
        # dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'visuals'))
        # self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        self.dataloader = self.config['dataloader']

        # model preparation
        self.E = self.E.to(device)
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        self.E.train()
        self.G.train()
        self.D.train()
    
    def gp(self, real_imgs, fake_imgs, att):
        bs = real_imgs.size(0)

        alpha = torch.rand(bs, 1)
        alpha = alpha.expand(real_imgs.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_imgs + ((1 - alpha) * fake_imgs)
        interpolates = interpolates.to(device)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.D(interpolates)
        # disc_interpolates = self.D(interpolates, att)

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.config['LAMBDA']
        return gradient_penalty

        """
        Implement gradient penalty function
        """
        pass

    def loss_cvae(self, recon_x, x, mu, logvar, criterion, se):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        se: semantic embedding
        """
        mse = criterion(recon_x, x)
        # KLD_element = (mu - se).pow(2).add_(logvar.exp()).mul_(-1).add(1).add_(logvar)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add(1).add_(logvar)

        KLD = torch.sum(KLD_element).mul_(-0.5)
        return mse + KLD
    
    def loss_cvae(self, recon_x, x, mu, logvar, criterion, se):
        """
        recon_x: generating images
        x: origin images
        mu: latent mean
        logvar: latent log variance
        se: semantic embedding
        """
        mse = criterion(recon_x, x)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add(1).add_(logvar)

        KLD = torch.sum(KLD_element).mul_(-0.5)
        return mse + KLD


    def train(self):
        self.prepare_environment()

        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            for i, (data, att) in enumerate(progress_bar):
                imgs = data.to(device)
                att = att.float().to(device)
                bs = imgs.size(0)
                
                # ***********
                # * Train D *
                # ***********
                # input_visual = Variable(torch.randn(bs, self.config['visual_dim'])).to(device)
                # input_att = Variable(torch.randn(bs, self.config['attr_dim'])).to(device)

                # cvae
                z = Variable(torch.randn(bs, self.config['visual_dim'] + self.config['attr_dim'])).to(device)

                # r_imgs = Variable(torch.cat((imgs, att), dim=1)).to(device)
                r_imgs = Variable(imgs).to(device)
                att = Variable(att).to(device)
                f_z, _, _ = self.E(imgs, att)
                f_imgs = self.G(f_z, att)
                r_label = torch.ones((bs)).unsqueeze(dim=1).to(device)
                f_label = torch.zeros((bs)).unsqueeze(dim=1).to(device)

                # Discriminator forwarding
                # r_logit = self.D(r_imgs, att)
                # f_logit = self.D(f_imgs, att)
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)
                
                """
                NOTE FOR SETTING DISCRIMINATOR LOSS:
                
                GAN: 
                    loss_D = (r_loss + f_loss)/2
                WGAN: 
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit)
                WGAN-GP: 
                    gradient_penalty = self.gp(r_imgs, f_imgs)
                    loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                """
                # Loss for discriminator
                # r_loss = self.loss(r_logit, r_label)
                # f_loss = self.loss(f_logit, f_label)
                # loss_D = (r_loss + f_loss) / 2
                
                # wgan-gp
                gradient_penalty = self.gp(r_imgs, f_imgs, att)                
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty

                # Discriminator backwarding
                self.D.zero_grad()
                loss_D.backward()
                self.opt_D.step()

                """
                Note FOR SETTING WEIGHT CLIP:

                WGAN: below code
                """
                # for p in self.D.parameters():
                #     p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

                # ***********
                # * Train G *
                # ***********
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images
                    # z = Variable(torch.randn(bs, self.config["z_dim"])).to(device)
                    z, mu, logvar = self.E(r_imgs, att)
                    f_imgs = self.G(z, att)
                    loss_E = self.loss_cvae(f_imgs, r_imgs, mu, logvar, self.mse, att)
                    
                    # Generator forwarding
                    # f_logit = self.D(f_imgs, att)
                    f_logit = self.D(f_imgs)

                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # Loss for the generator.
                    loss_G = self.loss(f_logit, r_label)

                    # Generator backwarding
                    self.G.zero_grad()
                    loss_G.backward(retain_graph=True)
                    self.E.zero_grad()
                    loss_E.backward()

                    self.opt_G.step()
                    self.opt_E.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_E=loss_E.item())
                self.steps += 1
            
            # self.G.eval()
            # f_imgs_sample = (self.G(self.z_samples).data + 1) / 2.0
            # filename = os.path.join(self.log_dir, f'Epoch_{epoch+1:03d}.jpg')
            # torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            # logging.info(f'Save some sample to {filename}.')

            if (e+1) % 5 == 0 or e == 0:
                # save the checkpoints.
                torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E_{e}.pth'))
                torch.save(self.G.state_dict(), os.path.join(self.ckpt_dir, f'G_{e}.pth'))
                torch.save(self.D.state_dict(), os.path.join(self.ckpt_dir, f'D_{e}.pth'))
            
        logging.info('Finish training')
   
    def inference(self, E_path, data, attr):
        self.E.load_state_dict(torch.load(E_path))
        self.E.to(device)
        self.E.eval()
        _, mu, _ = self.E(data, attr)
        return mu



class Discriminator(nn.Module):
    """
    NOTE FOR SETTING DISCRIMINATOR:

    You can't use nn.Batchnorm for WGAN-GP
    Use nn.InstanceNorm2d instead
    """
    def __init__(self, in_dim=85, feature_dim=2048):
        super(Discriminator, self).__init__()

        hiddens = [512, 128]

        self.l1 = nn.Sequential(
            nn.Linear(in_dim, hiddens[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(hiddens[1], hiddens[0]),
            nn.BatchNorm1d(hiddens[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(hiddens[0], feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.Sigmoid()
        )

        self.apply(weight_init)
    def forward(self, x):
        y = self.l1(x)
        return y

