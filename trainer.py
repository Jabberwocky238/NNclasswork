import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, utils
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import torch.nn.functional as F
import shutil
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO)

from vit import ViT
from swin_transformer_v2 import SwinTransformerV2
from time import perf_counter
# tensorboard --logdir=logs  
class Trainer:
    def __init__(self, task_name, model: ViT, optimizer, criterion, dataset='cifar100', root="/tmp/code/data", batch=128, start_epoch=0) -> None:
        self.root = root
        self.logger = logging.getLogger(task_name)
        self.logger.propagate = False
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(asctime)s] - %(name)s: %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        # print(self.logger.handlers)

        self.configure_dataset(dataset)
        self.train_dataloader = DataLoader(dataset=self.training_data, batch_size=batch, shuffle=True)
        self.test_dataloader = DataLoader(dataset=self.testing_data, batch_size=batch, shuffle=True)
        
        self.task_name = task_name
        self.writer = SummaryWriter(f'logs/{task_name}') 
        self.global_step = int((len(self.train_dataloader) + 1) / 30) * start_epoch

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # 定义学习率衰减策略
        self.optimizer = optimizer
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[5], gamma=0.1)
        self.criterion = criterion
        
    def configure_dataset(self, dataset_name):
        if dataset_name == 'cifar10':
            transform = transforms.Compose([
                # transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
            ])
            self.training_data = datasets.CIFAR10(root=self.root, train=True, download=True, transform=transform)
            self.testing_data = datasets.CIFAR10(root=self.root, train=False, download=True, transform=transform)
            self.classes = 10
        elif dataset_name == 'cifar100':
            transform = transforms.Compose([
                # transforms.Resize(size=(64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
            ])
            self.training_data = datasets.CIFAR100(root=self.root, train=True, download=False, transform=transform)
            self.testing_data = datasets.CIFAR100(root=self.root, train=False, download=False, transform=transform)
            self.classes = 100
        else:
            raise Exception("数据集设置有误")
        
        
    def train_step(self, epoch):
        self.model.train()
        time1 = perf_counter()
        running_loss = 0
        running_correct = 0

        for batch_idx, (X_train, y_train) in enumerate(self.train_dataloader):
            # X_train,y_train = torch.autograd.Variable(X_train),torch.autograd.Variable(y_train)
            X_train, y_train = X_train.to(self.device), y_train.to(self.device)
            outputs = self.model(X_train)
            _, pred = torch.max(outputs.data, 1)
            self.optimizer.zero_grad()
            loss = self.criterion(outputs, y_train)
            loss.backward()
            self.optimizer.step()

            running_loss = loss.item()
            running_correct = torch.sum(pred == y_train.data) / len(pred)
            if(batch_idx+1) % 30 == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(X_train), len(self.train_dataloader.dataset),
                    100. * batch_idx / len(self.train_dataloader), loss.item()))
                # print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     self.task_name,
                #     epoch, batch_idx * len(X_train), len(self.train_dataloader.dataset),
                #     100. * batch_idx / len(self.train_dataloader), loss.item()))
                self.writer.add_scalar('train/loss', running_loss, self.global_step)
                self.writer.add_scalar('train/mem_use', self.check_cuda_mem(), self.global_step)
                self.writer.add_scalar('train/running_correct', running_correct, self.global_step)
                self.global_step += 1
        self.writer.add_scalar('train/train_time_cost', perf_counter() - time1, epoch)
        self.lr_scheduler.step()
                    
    def test_step(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item() # 将一批的损失相加
                pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        self.logger.info(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_dataloader.dataset),
            100. * correct / len(self.test_dataloader.dataset)))
        # print('\n[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     self.task_name,
        #     test_loss, correct, len(self.test_dataloader.dataset),
        #     100. * correct / len(self.test_dataloader.dataset)))
        self.writer.add_scalar('test/loss', test_loss, epoch)
        self.writer.add_scalar('test/Accuracy', 100. * correct / len(self.test_dataloader.dataset), epoch)

    def save(self, epoch):
        import os
        if not os.path.isdir(f'./models/{self.task_name}/'):
            os.mkdir(f'./models/{self.task_name}/')
        torch.save(self.model.state_dict(), f'./models/{self.task_name}/{epoch}.pth')

    def check_cuda_mem(self):
        # 获取当前使用的GPU索引
        current_gpu_index = torch.cuda.current_device()
        # 获取GPU显存的总量和已使用量
        used_memory = torch.cuda.memory_allocated(current_gpu_index) / (1024 ** 3)  # 已使用显存(GB)
        return used_memory


if __name__ == '__main__':
    model_ViT = ViT(
        image_size = 32, patch_size = 4, num_classes = 100,
        dim = 512, depth = 6, heads = 8, 
        mlp_dim = 1024, norm_method='BN', dropout = 0.1, emb_dropout = 0.1
    )
    model_STV2 = SwinTransformerV2(
        img_size=32, patch_size=4, in_chans=3, num_classes=100,
        embed_dim=512, depths=[6], num_heads=[8],
        window_size=4, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer='BN', ape=False, patch_norm=True,
        use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
    )
    # model_ViT.load_state_dict(torch.load('models/a_BN_cifar100_ViT/0.pth'))
    # model_STV2.load_state_dict(torch.load('models/a_BN_cifar100_STV2/0.pth'))
    start_epoch = 0
    lr = 1e-3
    batch = 128
    root = './data'

    optimizer_ViT = torch.optim.Adam(model_ViT.parameters(), lr)
    optimizer_STV2 = torch.optim.Adam(model_STV2.parameters(), lr)
    criterion = torch.nn.CrossEntropyLoss()

    taskName = 'a_BN_cifar100_'
    
    trainer_ViT = Trainer(taskName+'ViT', model_ViT, optimizer_ViT, criterion, 
                          'cifar100', root, batch, start_epoch)
    trainer_STV2 = Trainer(taskName+'STV2', model_STV2, optimizer_STV2, criterion, 
                           'cifar100', root, batch, start_epoch)
    epochs = 10
    for epoch in range(start_epoch, epochs):
        logging.info("Epoch {}/{}".format(epoch+1, epochs))
        trainer_STV2.train_step(epoch)
        trainer_STV2.test_step(epoch)
        trainer_STV2.save(epoch)
        trainer_ViT.train_step(epoch)
        trainer_ViT.test_step(epoch)
        trainer_ViT.save(epoch)