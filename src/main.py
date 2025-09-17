from models import Discriminator, Generator
from engine import Trainer
from torch.optim import AdamW
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

def main(load_file=None, file_name=None):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.49139968, 0.48215827 ,0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
                                    ])

    # data = CIFAR10('~/dataset', train=True, transform=transform)
    data = ImageFolder(root='src/data', transform=transform)
    dataloader = DataLoader(dataset=data, batch_size=32, shuffle=True, pin_memory=True, drop_last=True)

    discriminator = Discriminator('discriminator.yaml')
    generator = Generator('generator.yaml')

    disc_opt = AdamW(discriminator.parameters(), lr=2e-4, betas=(0.9, 0.99))
    gen_opt = AdamW(generator.parameters(), lr=2e-4, betas=(0.9, 0.99))

    trainer = Trainer(discriminator=discriminator,
                    generator=generator,
                    optimizer_D=disc_opt,
                    optimizer_G=gen_opt,
                    data=dataloader,
                    writer_path='new_arch')
    if load_file:   
        trainer.load_checkpoint(f'runs/{file_name}' if file_name else f'runs/last')
    trainer.train(100)
    trainer.save_checkpoint(file_name if file_name else 'last')

def check_output_shape():
    BATCH = 32
    IMG_SIZE = [3, 64, 64]
    NOISE_SIZE = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(BATCH, IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    y = Discriminator('discriminator.yaml').to(device)(x.to(device))
    print(y.shape)

    x = torch.randn(BATCH, NOISE_SIZE, 1, 1).to(device)
    y = Generator('generator.yaml').to(device)(device)
    print(y.shape)

if __name__=='__main__':
    check_output_shape()
    main()