import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 48

        self.conv0 = nn.Conv1d(1, 48, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(48)
        self.stage0 = self._make_layer(block, 48, num_blocks[0], stride=1)
        self.stage1 = self._make_layer(block, 96, num_blocks[1], stride=2)
        self.stage2 = self._make_layer(block, 192, num_blocks[2], stride=2)
        self.stage3 = self._make_layer(block, 384, num_blocks[3], stride=2)
        self.fc = nn.Linear(384*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        
        out = self.conv0(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.avg_pool1d(out, 1)
        
        features = out.mean(dim=2)
        out = self.fc(features)
        
        return out, features

class ResNetSwav(nn.Module):
    def __init__(self, block, num_blocks,
            normalize=False, 
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,):
        super(ResNetSwav, self).__init__()
        self.in_planes = 64

        self.conv0 = nn.Conv1d(1, 64, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.stage0 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage1 = self._make_layer(block, 64 * 2, num_blocks[1], stride=2)
        self.stage2 = self._make_layer(block, 64 * 2 * 2, num_blocks[2], stride=2)
        self.stage3 = self._make_layer(block, 64 * 2 * 2 * 2, num_blocks[3], stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(64 * 2 * 2 * 2 * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(64 * 2 * 2 * 2 * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)


    def forward_backbone(self, x):
        
        out = self.conv0(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.max_pool1d(out, 1)
        out = self.stage0(out)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.adaptive_pool(out)
        out = torch.flatten(out, 1)
        
        return out
    
    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x, None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx: end_idx])

            if 'cuda' in str(self.conv0.weight.device):
                _out = self.forward_backbone(_out.cuda(non_blocking=True))
            else:
                _out = self.forward_backbone(_out)

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)


def ResNet18(num_classes=2):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=2):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=2):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=2):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=2):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)

def ResNetSwav50(output_dim, hidden_mlp, nmb_prototypes):
    return ResNetSwav(Bottleneck, [3,4,6,3], 
            normalize=False, 
            output_dim=output_dim,
            hidden_mlp=hidden_mlp,
            nmb_prototypes=nmb_prototypes)

def ResNetSwav101(output_dim, hidden_mlp, nmb_prototypes):
    return ResNetSwav(Bottleneck, [3,4,23,3], 
            normalize=False, 
            output_dim=output_dim,
            hidden_mlp=hidden_mlp,
            nmb_prototypes=nmb_prototypes)

def ResNetSwav152(output_dim, hidden_mlp, nmb_prototypes):
    return ResNetSwav(Bottleneck, [3,8,36,3], 
            normalize=False, 
            output_dim=output_dim,
            hidden_mlp=hidden_mlp,
            nmb_prototypes=nmb_prototypes)

# def test():
#     net = ResNet18()
#     y = net(torch.randn(3,1,1200))
#     print(y[0].size(), y[1].size())

# test()