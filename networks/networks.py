import torch.nn as nn
import torch.nn.functional as F

from . import utils


class LeNet(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out


class AlexCifarNet(utils.ReparamModule):
    supported_dims = {32}

    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(utils.ReparamModule):
    supported_dims = {224}

    class Idt(nn.Module):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


class ResNet18SimCLR(utils.ReparamModule):
    supported_dims = {32, 224}

    def __init__(self, state):
        super(ResNet18SimCLR, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        
        from simclr import SimCLR
        from simclr.modules import get_resnet
        
        # initialize ResNet
        encoder = get_resnet("resnet18", pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        self.features = SimCLR(encoder, 64, n_features)
        
        from simclr.modules import NT_Xent
        import torchvision.transforms as transforms 
        
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        _train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=state.input_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        _test_transforms = transforms.Resize(state.input_size)
        
        _lambda_train = lambda x : (_train_transforms(x), _train_transforms(x))
        _lambda_test = lambda x : (_test_transforms(x), _test_transforms(x))
        
        self.train_transforms = transforms.Compose([
            transforms.Lambda(_lambda_train)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Lambda(_lambda_test)])
        
        self.criterion_distill = NT_Xent(state.num_classes*state.distilled_images_per_class_per_step, 0.5, state.world_size)
        self.criterion_real = NT_Xent(state.batch_size, 0.5, state.world_size)        
        
    def forward(self, x):
        if self.training:
            x_i, x_j = self.train_transforms(x)
            h_i, h_j, z_i, z_j = self.features(x_i, x_j)
            loss = self.criterion_distill(z_i, z_j)
        else:
            x_i, x_j = self.test_transforms(x)
            h_i, h_j, z_i, z_j = self.features(x_i, x_j)
            loss = self.criterion_real(z_i, z_j)
        
        return loss
    
class ResNet50SimCLR(utils.ReparamModule):
    supported_dims = {32, 224}

    def __init__(self, state):
        super(ResNet18SimCLR, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        
        from simclr import SimCLR
        from simclr.modules import get_resnet
        
        # initialize ResNet
        encoder = get_resnet("resnet50", pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        self.features = SimCLR(encoder, 64, n_features)
        
        from simclr.modules import NT_Xent
        import torchvision.transforms as transforms 
        
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        _train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=state.input_size),
            transforms.RandomHorizontalFlip(),  # with 0.5 probability
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        _test_transforms = transforms.Resize(state.input_size)
        
        _lambda_train = lambda x : (_train_transforms(x), _train_transforms(x))
        _lambda_test = lambda x : (_test_transforms(x), _test_transforms(x))
        
        self.train_transforms = transforms.Compose([
            transforms.Lambda(_lambda_train)
        ])
        self.test_transforms = transforms.Compose([
            transforms.Lambda(_lambda_test)])
        
        self.criterion_distill = NT_Xent(state.num_classes*state.distilled_images_per_class_per_step, 0.5, state.world_size)
        self.criterion_real = NT_Xent(state.batch_size, 0.5, state.world_size)
        
    def forward(self, x):
        if self.training:
            x_i, x_j = self.train_transforms(x)
            h_i, h_j, z_i, z_j = self.features(x_i, x_j)
            loss = self.criterion_distill(z_i, z_j)
        else:
            x_i, x_j = self.test_transforms(x)
            h_i, h_j, z_i, z_j = self.features(x_i, x_j)
            loss = self.criterion_real(z_i, z_j)
        
        return loss