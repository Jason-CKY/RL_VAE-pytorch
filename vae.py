import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder

class VAE(nn.Module):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=128, device='cpu'):
        super().__init__()

        self.latent_dim = latent_dim
        self.device = device
        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim, 
            input_height=input_height, 
            first_conv=False, 
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def gaussian_likelihood(self, mean, logscale, sample):
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def encode_image(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        return z

    def get_elbo_loss(self, x):
        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample().cuda()

        # decoded 
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        log_dict = {
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean()
        }

        return elbo, log_dict

    def reconstruct(self, n_preds):
        '''
        Decode from a normal distribution to give images
        '''

        # Z COMES FROM NORMAL(0, 1)
        num_preds = n_preds
        p = torch.distributions.Normal(torch.zeros((self.latent_dim,)), torch.ones((self.latent_dim,)))
        z = p.rsample((num_preds,))

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = self.decoder(z.to(self.device)).cpu()

        return pred

    def save_weights(self, fpath):
        print('saving checkpoint...')
        checkpoint = {
            'encoder': self.encoder.state_dict(),
            'fc_mu': self.fc_mu.state_dict(),
            'fc_var': self.fc_var.state_dict(),
            'decoder': self.decoder.state_dict()
        }
        torch.save(checkpoint, fpath)
        print(f"checkpoint saved at {fpath}")    
    
    def load_weights(self, fpath):
        if os.path.isfile(fpath):
            checkpoint = torch.load(fpath, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.fc_mu.load_state_dict(checkpoint['fc_mu'])
            self.fc_var.load_state_dict(checkpoint['fc_var'])
            self.decoder.load_state_dict(checkpoint['decoder'])

            print('checkpoint loaded at {}'.format(fpath))
        else:
            raise AssertionError(f"No weights file found at {fpath}")

    def dataparallel(self, ngpu):
        print(f"using {ngpu} gpus, gpu id: {list(range(ngpu))}")
        self.encoder = nn.DataParallel(self.encoder, list(range(ngpu)))
        self.decoder = nn.DataParallel(self.decoder, list(range(ngpu)))
        self.fc_mu = nn.DataParallel(self.fc_mu, list(range(ngpu)))
        self.fc_var = nn.DataParallel(self.fc_var, list(range(ngpu)))