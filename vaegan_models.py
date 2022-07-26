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
from torch.utils import data
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import optim
from torch.autograd import Variable, grad
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm
import others.data_utils as data_utils
import others.classifier as classifier

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# setting for weight init function
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        hidden_dims = config['enc_hidden_dims'].copy()
        latent_dim = config['latent_dim']
        
        if self.config['model_type'] == 'cvae' or self.config['model_type'] == 'cvaegan':
            in_dim = hidden_dims[0] + config['latent_dim']
        else:
            in_dim = hidden_dims[0]

        self.fc1 = nn.Linear(in_dim, hidden_dims[-1])
        self.fc2 = nn.Linear(hidden_dims[-1], latent_dim*2)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.enc_mu = nn.Linear(latent_dim*2, latent_dim)
        self.enc_logvar = nn.Linear(latent_dim*2, latent_dim)

        self.apply(weight_init)
    
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def forward(self, x, se=None):
        if self.config['model_type'] == 'cvae' or self.config['model_type'] == 'cvaegan':
            x = torch.cat((x, se), dim=-1)

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

        self.config = config
        hidden_dims = config['dec_hidden_dims'].copy()
        latent_dim = config['latent_dim']

        if config['model_type'] == 'cvae' or config['model_type'] == 'cvaegan':
            in_dim = latent_dim * 2
        else:
            in_dim = latent_dim

        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)
    
    def forward(self, z, se=None):
        if self.config['model_type'] == 'cvae' or self.config['model_type'] == 'cvaegan':
            z = torch.cat((z, se), dim=-1)
            
        x = self.lrelu(self.fc1(z))
        x = self.sigmoid(self.fc2(x))
        return x
        

class AttDec(nn.Module):
    def __init__(self, config):
        super(AttDec, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(self.config['visual_dim'] + self.config['attr_dim'], 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, self.config['attr_dim']),
            nn.Sigmoid(),
        )
        self.apply(weight_init)
    
    def forward(self, x, att=None):
        if att is not None:
            h = torch.cat((x, att), dim=1)
        else:
            h = x
        
        output = self.model(h)
        return output


class Discriminator_D1(nn.Module):
    def __init__(self, config):
        super(Discriminator_D1, self).__init__()
        self.config = config
        
        if config['model_type'] == 'cvae' or config['model_type'] == 'cvaegan':
            in_dim = config['visual_dim'] + config['attr_dim']
        else:
            in_dim = config['visual_dim']

        self.fc1 = nn.Linear(in_dim, config['d_hdim'])
        self.fc2 = nn.Linear(config['d_hdim'], 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weight_init)
    def forward(self, x, att=None):
        if self.config['model_type'] == 'cvae' or self.config['model_type'] == 'cvaegan':
            h = torch.cat((x, att), dim=1)
        else:
            h = x
        self.hidden = self.lrelu(self.fc1(h))
        h = self.fc2(self.hidden)
        h = self.sigmoid(h)
        return h






def generate_syn_feature(config, generator, classes, attribute, num, netF=None, netDec=None):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, config['visual_dim'])
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, config['attr_dim'])
    syn_noise = torch.FloatTensor(num, config['attr_dim'])
    if config['cuda']:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        syn_noisev = Variable(syn_noise, requires_grad=True)
        syn_attv = Variable(syn_att, requires_grad=True)
        fake = generator(syn_noisev, syn_attv)
        output = fake
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)
    
    return syn_feature, syn_label



class TrainerGAN():
    def __init__(self, config):
        self.config = config

        self.E = Encoder(self.config)
        self.G = Generator(self.config)
        self.D = Discriminator_D1(self.config)
        self.attD = AttDec(self.config)

        self.loss = nn.BCELoss()
        self.mse = nn.MSELoss(reduction='mean')

        """
        NOTE FOR SETTING OPTIMIZER:

        GAN: use Adam optimizer
        WGAN: use RMSprop optimizer
        WGAN-GP: use Adam optimizer 
        """
        self.opt_E = torch.optim.Adam(self.E.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        self.opt_attD = torch.optim.Adam(self.attD.parameters(), lr=self.config["lr"], betas=(0.5, 0.999))
        # self.log_dir = os.path.join(self.config["workspace_dir"], 'logs')
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
        # os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # self.log_dir = os.path.join(self.log_dir, self.config['dataset'], time)
        self.ckpt_dir = os.path.join(self.ckpt_dir, self.config['dataset'], time)
        # os.makedirs(self.log_dir)
        os.makedirs(self.ckpt_dir)
        
        # create dataset by the above function
        # dataset = get_dataset(os.path.join(self.config["workspace_dir"], 'visuals'))
        # self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)
        self.dataloader = self.config['dataloader']

        # model preparation
        self.E = self.E.to(device)
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        self.attD = self.attD.to(device)
        self.E.train()
        self.G.train()
        self.D.train()
        self.attD.train()

        self.max_e_loss = np.inf
        self.tmp_e_loss = 0

        self.dataset = data_utils.DATA_LOADER(self.config)

    
    def gp(self, real_imgs, fake_imgs, att):
        bs = real_imgs.size(0)

        alpha = torch.rand(bs, 1)
        alpha = alpha.expand(real_imgs.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_imgs + ((1 - alpha) * fake_imgs)
        interpolates = interpolates.to(device)
        # interpolates = Variable(interpolates, requires_grad=True)
        interpolates = Variable(torch.cat((interpolates, att),dim=1), requires_grad=True)

        # disc_interpolates = self.D(interpolates)
        # disc_interpolates = self.D(interpolates, att)
        disc_interpolates = self.D(interpolates)
        # inter_att = torch.cat((interpolates, att), dim=1)

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
        KLD_element = (mu - se).pow(2).add_(logvar.exp()).mul_(-1).add(1).add_(logvar)
        # KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add(1).add_(logvar)

        KLD = torch.sum(KLD_element).mul_(-0.5)
        return mse + KLD
    
    def inference(self, E_path, data, attr=None):
        self.E.load_state_dict(torch.load(E_path))
        self.E.to(device)
        self.E.eval()
        if attr is not None:
            _, mu, _ = self.E(data, attr)
        else:
            _, mu, _ = self.E(data)
        return mu
    
    def train_cvae(self):
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            self.tmp_e_loss = 0

            for i, (data, att) in enumerate(progress_bar):
                imgs = data.to(device)
                att = att.float().to(device)
                bs = imgs.size(0)

                r_imgs = Variable(imgs).to(device)
                z, mu, logvar = self.E(r_imgs, att)
                f_imgs = self.G(z, att)
                loss_E = self.loss_cvae(f_imgs, r_imgs, mu, logvar, self.mse, att)
                self.tmp_e_loss += loss_E.item()

                self.E.zero_grad()
                self.G.zero_grad()
                loss_E.backward()
                self.opt_G.step()
                self.opt_E.step()
            
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_E=loss_E.item())
                self.steps += 1
        
            if self.tmp_e_loss < self.max_e_loss:
                # torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E_{e+1}.pth'))
                torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E.pth'))
                print(f"E_{e+1} better than privious")
                self.max_e_loss = self.tmp_e_loss
        logging.info('Finish training')
    

    def train_vae(self):
        self.prepare_environment()
        
        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            self.tmp_e_loss = 0

            for i, (data, att) in enumerate(progress_bar):
                imgs = data.to(device)
                att = att.float().to(device)
                bs = imgs.size(0)

                r_imgs = Variable(imgs).to(device)
                z, mu, logvar = self.E(r_imgs)
                f_imgs = self.G(z)
                loss_E = self.loss_cvae(f_imgs, r_imgs, mu, logvar, self.mse, att)
                self.tmp_e_loss += loss_E.item()

                self.E.zero_grad()
                self.G.zero_grad()
                loss_E.backward()
                self.opt_G.step()
                self.opt_E.step()
            
                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_E=loss_E.item())
                self.steps += 1
        
            if self.tmp_e_loss < self.max_e_loss:
                # torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E_{e+1}.pth'))
                torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E.pth'))
                print(f"E_{e+1} better than privious")
                self.max_e_loss = self.tmp_e_loss
        logging.info('Finish training')

        
    def train(self):
        self.prepare_environment()

        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            self.tmp_e_loss = 0
            
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
                # z = Variable(torch.randn(bs, self.config['visual_dim'] + self.config['attr_dim'])).to(device)
                n_v = Variable(torch.randn(bs, self.config['visual_dim'])).to(device)
                

                # r_imgs = Variable(torch.cat((imgs, att), dim=1)).to(device)
                r_imgs = Variable(imgs).to(device)
                att = Variable(att).to(device)
                f_z, _, _ = self.E(imgs, att)
                f_imgs = self.G(f_z, att)
                # n_imgs = self.G(n_v, att)
                # r_label = torch.ones((bs)).unsqueeze(dim=1).to(device)
                # f_label = torch.zeros((bs)).unsqueeze(dim=1).to(device)

                # Discriminator forwarding
                r_logit = self.D(r_imgs, att)
                f_logit = self.D(f_imgs, att)
                n_logit = self.D(n_v, att)
                
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
                
                # wgan-gp
                # gradient_penalty = self.gp(r_imgs, f_imgs, att)                
                # loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
                
                # wgan
                loss_D = -torch.mean(r_logit) + torch.mean(f_logit) + torch.mean(n_logit)

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
                    e_z, mu, logvar = self.E(r_imgs, att)
                    f_imgs = self.G(e_z, att)
                    loss_E = self.loss_cvae(f_imgs, r_imgs, mu, logvar, self.mse, att)
                    
                    self.tmp_e_loss += loss_E.item()
                    # Generator forwarding
                    f_logit = self.D(f_imgs, att)
                    # f_logit = self.D(f_imgs)

                    n_z = Variable(torch.randn(bs, self.config['attr_dim'])).to(device)
                    n_imgs = self.G(n_z, att)

                    """
                    NOTE FOR SETTING LOSS FOR GENERATOR:
                    
                    GAN: loss_G = self.loss(f_logit, r_label)
                    WGAN: loss_G = -torch.mean(self.D(f_imgs))
                    WGAN-GP: loss_G = -torch.mean(self.D(f_imgs))
                    """
                    # Loss for the generator.
                    # loss_G = self.loss(f_logit, r_label)

                    # WGAN
                    loss_G = -torch.mean(self.D(f_imgs, att)) - torch.mean(self.D(n_imgs, att))

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
            
            if self.tmp_e_loss < self.max_e_loss:
                # torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E_{e+1}.pth'))
                torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E.pth'))
                print(f"E_{e+1} better than privious")
                self.max_e_loss = self.tmp_e_loss

        logging.info('Finish training')
    
    def train_cycle_vaegan(self):
        self.prepare_environment()
        best_gzsl_acc = 0
        best_zsl_acc = 0

        for e, epoch in enumerate(range(self.config["epochs"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e+1}")
            self.tmp_e_loss = 0
            
            for i, (data, att) in enumerate(progress_bar):
                imgs = data.to(device)
                att = att.float().to(device)
                bs = imgs.size(0)
                
                # ***********
                # * Train D *
                # ***********
                z = Variable(torch.randn(bs, self.config['visual_dim'] + self.config['attr_dim'])).to(device)

                # r_imgs = Variable(torch.cat((imgs, att), dim=1)).to(device)
                r_imgs = Variable(imgs).to(device)
                att = Variable(att).to(device)
                f_z, _, _ = self.E(imgs, att)
                f_imgs = self.G(f_z, att)
                r_label = torch.ones((bs)).unsqueeze(dim=1).to(device)
                f_label = torch.zeros((bs)).unsqueeze(dim=1).to(device)

                # Discriminator forwarding
                r_logit = self.D(r_imgs)
                f_logit = self.D(f_imgs)
                
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
                # * Train VAE *
                # ***********
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images
                    # z = Variable(torch.randn(bs, self.config["z_dim"])).to(device)
                    z, mu, logvar = self.E(r_imgs, att)
                    f_imgs = self.G(z, att)
                    loss_E = self.loss_cvae(f_imgs, r_imgs, mu, logvar, self.mse, att)
                    
                    self.tmp_e_loss += loss_E.item()
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

                    # ***********
                    # * Train VAE *
                    # ***********
                    z2, mu2, logvar2 = self.E(f_imgs, att)
                    recons_imgs = self.G(z2, att)
                    recons_loss_E = self.loss_cvae(recons_imgs, f_imgs, mu2, logvar2, self.mse, att)

                    loss_E = recons_loss_E + loss_E
                    ce_bce_loss = self.mse(z, z2)

                    loss_G = loss_G + ce_bce_loss

                    # Generator backwarding
                    self.G.zero_grad()
                    self.E.zero_grad()
                    # ce_bce_loss.backward(retain_graph=True)
                    loss_G.backward(retain_graph=True)
                    # ce_bce_loss.backward(retain_graph=True)
                    loss_E.backward()

                    self.opt_G.step()
                    self.opt_E.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), loss_E=loss_E.item(), ce_bce=ce_bce_loss.item())
                self.steps += 1
            
            if self.tmp_e_loss < self.max_e_loss:
                torch.save(self.E.state_dict(), os.path.join(self.ckpt_dir, f'E.pth'))
                print(f'E_{e+1} better than previous')
                self.max_e_loss = self.tmp_e_loss
            
            self.G.eval()
            self.D.eval()
            syn_feature, syn_label = generate_syn_feature(self.config, self.G, self.dataset.unseenclasses, 
                                            self.dataset.attribute, self.config['syn_num'])
            
            if self.config['gzsl']:
                train_X = torch.cat((self.dataset.train_feature, syn_feature), 0)
                train_Y = torch.cat((self.dataset.train_label, syn_label), 0)

                nclass = self.config['class_num']
                gzsl_cls = classifier.CLASSIFIER(train_X, train_Y, self.dataset, nclass, self.config['cuda'], self.config['classifier_lr'], 
                                                0.5, 25, self.config['syn_num'], generalized=True)
                if best_gzsl_acc < gzsl_cls.H:
                    best_acc_seen, best_acc_unseen, best_gzsl_acc = gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H
                print('GZSL: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_cls.acc_seen, gzsl_cls.acc_unseen, gzsl_cls.H),end=" ")
            
            # zero shot learning
            # train zsl classifier

            if self.config['zsl']:
                zsl_cls = classifier.CLASSIFIER(syn_feature, data_utils.map_label(syn_label, self.dataset.unseenclasses),
                                            self.dataset, self.dataset.unseenclasses.size(0), self.config['cuda'],
                                            self.config['classifier_lr'], 0.5, 25, self.config['syn_num'],
                                            generalized=False)
                acc  = zsl_cls.acc
                if best_zsl_acc < acc:
                    best_zsl_acc = acc
                print('ZSL: unseen accuracy=%.4f' % (acc))

            self.G.train()
            self.D.train()
        
        print('Dataset', self.config['dataset'])
        print('the best ZSL unseen accuracy is', best_zsl_acc)
        if self.config['gzsl']:
            print('Dataset', self.config['dataset'])
            print('the best GZSL seen accuracy is', best_acc_seen)
            print('the best GZSL unseen accuracy is', best_acc_unseen)
            print('the best GZSL H is', best_gzsl_acc)
            
        logging.info('Finish training')


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

