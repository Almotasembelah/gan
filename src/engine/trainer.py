import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm.auto import tqdm

class Trainer:
    def __init__(self, 
                 discriminator:nn.Module, 
                 generator:nn.Module, 
                 optimizer_D:torch.optim.Optimizer, 
                 optimizer_G:torch.optim.Optimizer, 
                 data=None, 
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 writer_path='logs'):
        self.device = device
        self.discriminator = discriminator.to(self.device)
        self.generator = generator.to(self.device)
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G
        self.data = data

        self.loss_fn = nn.BCELoss()
        self.MSE = nn.MSELoss()

        self.real_writer = SummaryWriter(f'logs/{writer_path}/real')
        self.fake_writer = SummaryWriter(f'logs/{writer_path}/fake')
        self.step = 0
        self.total_epochs = 0

    def set_optimizers(self, optimizer_D, optimizer_G):
        self.optimizer_D = optimizer_D
        self.optimizer_G = optimizer_G

    def set_data(self, data):
        self.data = data 
    
    def train_discriminator(self, real, fake):
        real_pred = self.discriminator(real).reshape(-1)
        fake_pred = self.discriminator(fake).reshape(-1)
        loss_real = self.loss_fn(real_pred, torch.ones_like(real_pred))
        loss_fake = self.loss_fn(fake_pred, torch.zeros_like(fake_pred))

        loss = (loss_real + loss_fake)/2
        self.optimizer_D.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_D.step()
        return loss.item()

    def train_generator(self, fake, real):
        self.optimizer_G.zero_grad()
        fake_pred = self.discriminator(fake).reshape(-1)
        loss = self.loss_fn(fake_pred, torch.ones_like(fake_pred))
        loss.backward()
        self.optimizer_G.step()
        return loss.item()

    def train(self, epochs):
        self.discriminator.train()
        self.generator.train()
        for epoch in tqdm(range(epochs)):
            self.total_epochs += 1
            d_loss, g_loss = [], []
            for i, (real, _) in tqdm(enumerate(self.data)):
                real = real.to(self.device)
                fake = self.generator(self.device)
                disc_loss = self.train_discriminator(real=real, fake=fake)
                gen_loss = self.train_generator(fake=fake, real=real)

                d_loss.append(disc_loss)
                g_loss.append(gen_loss)
                if i % 10 == 0:
                    print(f'Epoch {epoch+1} | Step: {self.step} | Discriminator Loss: {disc_loss:.4f} | Generator Loss {gen_loss:0.4f}')
                    self.val(real)
                    self.step += 1
            self.fake_writer.add_scalar('Loss/Discriminator', sum(d_loss)/len(d_loss), self.total_epochs)
            self.fake_writer.add_scalar('Loss/Generator', sum(g_loss)/len(g_loss), self.total_epochs)

        self.real_writer.flush()
        self.fake_writer.flush()
        self.real_writer.close()
        self.fake_writer.close()

    @torch.no_grad()
    def val(self, real, batch=32):
        self.generator.eval()
        fake = self.generator(self.device)

        real_grid = make_grid(real[:batch], normalize=True)
        fake_grid = make_grid(fake[:batch], normalize=True)

        self.real_writer.add_image('Real', real_grid, global_step=self.step)
        self.fake_writer.add_image('Fake', fake_grid, global_step=self.step)
        self.generator.train()

    def save_checkpoint(self, filename):
        import os

        # Create runs folder if it doesn't exist
        runs_dir = "runs"
        os.makedirs(runs_dir, exist_ok=True)

        # Full path for the checkpoint file
        checkpoint_path = os.path.join(runs_dir, filename)

        checkpoint = {
                    'disc_state_dict': self.discriminator.state_dict(),
                    'gen_state_dict': self.generator.state_dict(),
                    'optimizer_d_state_dict': self.optimizer_D.state_dict(),
                    'optimizer_g_state_dict': self.optimizer_G.state_dict(),
                    'writer_step':self.step,
                    'total_epochs':self.total_epochs
                    }
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, filename):
        """
        Loads the model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=torch.device(self.device))
        self.discriminator.load_state_dict(checkpoint['disc_state_dict'])
        self.generator.load_state_dict(checkpoint['gen_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.step = checkpoint['writer_step']
        self.total_epochs = checkpoint['total_epochs']
        self.discriminator.train()
        self.generator.train()