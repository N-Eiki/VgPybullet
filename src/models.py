#!/usr/bin/env python

from collections import OrderedDict
import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import time
import cv2


class reactive_net(nn.Module):

    def __init__(self, use_cuda,): # , snapshot=None
        super(reactive_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.push_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.push_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(pretrained=True)
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(pretrained=True)

        self.num_rotations = 16

        # Construct network branches for pushing and grasping
        self.pushnet = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(2048)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('push-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(64, 3, kernel_size=1, stride=1, bias=False))
            # ('grasp-upsample2', nn.Upsample(scale_factor=4, mode='bilinear'))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            output_prob = []
            interm_feat = []

            # Apply rotations to images
            for rotate_idx in range(self.num_rotations):
                rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

                # Compute sample grid for rotation BEFORE neural network
                affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                affine_mat_before.shape = (2,3,1)
                affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                else:
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                # Rotate images clockwise
                if self.use_cuda:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                else:
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                # Compute intermediate features
                interm_push_color_feat = self.push_color_trunk.features(rotate_color)
                interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
                interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
                interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
                interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
                interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
                interm_feat.append([interm_push_feat, interm_grasp_feat])

                # Compute sample grid for rotation AFTER branches
                affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                affine_mat_after.shape = (2,3,1)
                affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                if self.use_cuda:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
                else:
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())

                # Forward pass through branches, undo rotation on output predictions, upsample results
                output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                    nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.push_color_trunk.features(rotate_color)
            interm_push_depth_feat = self.push_depth_trunk.features(rotate_depth)
            interm_push_feat = torch.cat((interm_push_color_feat, interm_push_depth_feat), dim=1)
            interm_grasp_color_feat = self.grasp_color_trunk.features(rotate_color)
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(rotate_depth)
            interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
            self.interm_feat.append([interm_push_feat, interm_grasp_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), interm_push_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), interm_push_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.pushnet(interm_push_feat), flow_grid_after, mode='nearest')),
                                     nn.Upsample(scale_factor=16, mode='bilinear').forward(F.grid_sample(self.graspnet(interm_grasp_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat


class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_num, identity_conv=None, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_num, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=kernel_num, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.identity_conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_num, stride=stride, padding=1)
    def forward(self, x):
        identity = x.clone()  # save the input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_conv is not None:
            identity = self.identity_conv(identity)
        x += identity
        x = self.relu(x)
        return x

class reinforcement_net(nn.Module):

    def __init__(self, use_cuda, frames=1): # , snapshot=None
        super(reinforcement_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.num_rotations = 16

        self.perceptnet = nn.Sequential(OrderedDict([
            ('percept-conv0', nn.Conv2d(4*frames, 64, kernel_size=3, stride=1, bias=False)),
            ('percept-maxpool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('percept-resblock0', ResBlock(64, 128, 3)),
            ('percept-maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('percept-resblock1', ResBlock(128, 256, 3)),
            ('percept-resblock2', ResBlock(256, 512, 3))
        ]))

        
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-resblock0', ResBlock(640, 256, 3)),
            ('grasp-resblock1', ResBlock(256, 128, 3)),
            ('grasp-upsample0', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('grasp-resblock2', ResBlock(128, 64, 3)),
            ('grasp-upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('grasp-conv0', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))


        self.regulatenet = nn.Sequential(OrderedDict([
            ('regulate-resblock0', ResBlock(640, 256, 3)),
            ('regulate-resblock1', ResBlock(256, 128, 3)),
            ('regulate-upsample0', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('thrregulateow-resblock2', ResBlock(128, 64, 3)),
            ('regulate-upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('regulate-conv0', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))
        
        #  self.graspnet_target = nn.Sequential(OrderedDict([
        #     ('grasp-resblock0', ResBlock(640, 256, 3)),
        #     ('grasp-resblock1', ResBlock(256, 128, 3)),
        #     ('grasp-upsample0', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        #     ('grasp-resblock2', ResBlock(128, 64, 3)),
        #     ('grasp-upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        #     ('grasp-conv0', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        # ]))


        # self.regulatenet_target = nn.Sequential(OrderedDict([
        #     ('regulate-resblock0', ResBlock(640, 256, 3)),
        #     ('regulate-resblock1', ResBlock(256, 128, 3)),
        #     ('regulate-upsample0', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        #     ('thrregulateow-resblock2', ResBlock(128, 64, 3)),
        #     ('regulate-upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
        #     ('regulate-conv0', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        # ]))
        
        # Initialize network weights
        for m in self.named_modules():
            if 'regulate-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()
        self.regulatenet.state_dict()["regulate-conv0.weight"] = torch.nn.init.zeros_(self.regulatenet.state_dict()["regulate-conv0.weight"])

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def sync_net(self,):
        self.graspnet_target = copy.deepcopy(self.graspnet)
        self.regulatenet_target = copy.deepcopy(self.regulatenet)

    def forward_one_scene(self, input_color_data, input_depth_data, rotate_theta):
        affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
        affine_mat_before.shape = (2,3,1)
        affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
        if self.use_cuda:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
        else:
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

        # Rotate images clockwise
        if self.use_cuda:
            rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
            rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
        else:
            rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
            rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

        # Compute intermediate features
        rotate_data = torch.cat([rotate_color, rotate_depth], dim=1)
        return rotate_data

    def forward(self, inputs_color_data, inputs_depth_data, is_volatile=False, specific_rotation=-1, base_velocity=None):
        # input_color_data = nn.AvgPool2d((2,2))(input_color_data)
        # input_depth_data = nn.AvgPool2d((2,2))(input_depth_data)
        inputs_color_data = [nn.AvgPool2d((2,2))(input_color_data) for input_color_data in inputs_color_data]
        inputs_depth_data = [nn.AvgPool2d((2,2))(input_depth_data) for input_depth_data in inputs_depth_data]

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    # Compute sample grid for rotation BEFORE neural network
                    rotate_data = []
                    for input_color_data, input_depth_data in zip(inputs_color_data, inputs_depth_data):
                        rotate_data.append(self.forward_one_scene(input_color_data, input_depth_data, rotate_theta))
                    rotate_data = torch.cat(rotate_data, axis=1)
                    # Compute intermediate features
                    interm_percept_feat = self.perceptnet(rotate_data.cuda())
                    input_velocity_data = torch.full((1, 128, 80, 80), fill_value=base_velocity).cuda()
                    interm_percept_feat = torch.cat([interm_percept_feat, input_velocity_data], dim=1)
                    
                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.data.size(), align_corners=True)
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), input_color_data.data.size(), align_corners=True)

                    # Forward pass through branches, undo rotation on output predictions results

                    regulate_prop = F.grid_sample(self.regulatenet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True)
                    output_prob.append([
                        regulate_prop,
                        F.grid_sample(self.graspnet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True),]
                                        )

            return output_prob, None

        else:
            self.output_prob = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            rotate_data = []
            for input_color_data, input_depth_data in zip(inputs_color_data, inputs_depth_data):
                rotate_data.append(self.forward_one_scene(input_color_data, input_depth_data, rotate_theta))
            rotate_data = torch.cat(rotate_data, axis=1)
            # # Compute intermediate features
            # rotate_data = torch.cat([rotate_color, rotate_depth], dim=1)
            interm_percept_feat = self.perceptnet(rotate_data.cuda())
            input_velocity_data = torch.full((1, 128, 80, 80), fill_value=base_velocity).cuda()
            interm_percept_feat = torch.cat([interm_percept_feat,input_velocity_data], dim=1)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.data.size(), align_corners=True)
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), input_color_data.data.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions results
            regulate_prop = F.grid_sample(self.regulatenet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True)
                    
            self.output_prob.append([
                                    regulate_prop,
                                     F.grid_sample(self.graspnet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True),
            ])

           
            return self.output_prob, None

class reinforcement_static_net(nn.Module):

    def __init__(self, use_cuda): # , snapshot=None
        super(reinforcement_static_net, self).__init__()
        self.use_cuda = use_cuda

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.num_rotations = 16

        self.perceptnet = nn.Sequential(OrderedDict([
            ('percept-conv0', nn.Conv2d(4, 64, kernel_size=3, stride=1, bias=False)),
            ('percept-maxpool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('percept-resblock0', ResBlock(64, 128, 3)),
            ('percept-maxpool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ('percept-resblock1', ResBlock(128, 256, 3)),
            ('percept-resblock2', ResBlock(256, 512, 3))
        ]))

        
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-resblock0', ResBlock(512, 256, 3)),
            ('grasp-resblock1', ResBlock(256, 128, 3)),
            ('grasp-upsample0', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('grasp-resblock2', ResBlock(128, 64, 3)),
            ('grasp-upsample1', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)),
            ('grasp-conv0', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        
        # Initialize network weights
        for m in self.named_modules():
            if 'regulate-' in m[0] or 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []


    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1, ):
        input_color_data = nn.AvgPool2d((2,2))(input_color_data)
        input_depth_data = nn.AvgPool2d((2,2))(input_depth_data)

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest')
                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before, mode='nearest')
                        rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True), flow_grid_before, mode='nearest')

                    # Compute intermediate features
                    rotate_data = torch.cat([rotate_color, rotate_depth], dim=1)

                    # Compute intermediate features
                    interm_percept_feat = self.perceptnet(rotate_data.cuda())
                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.data.size(), align_corners=True)
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), input_color_data.data.size(), align_corners=True)

                    # Forward pass through branches, undo rotation on output predictions results

                    output_prob.append([
                        None,
                        F.grid_sample(self.graspnet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True),]
                                        )

            return output_prob, None

        else:
            self.output_prob = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False), input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before, mode='nearest')
                rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False), flow_grid_before, mode='nearest')

            # Compute intermediate features
            
            rotate_data = torch.cat([rotate_color, rotate_depth], dim=1)
            interm_percept_feat = self.perceptnet(rotate_data.cuda())

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.data.size(), align_corners=True)
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False), input_color_data.data.size(), align_corners=True)

            # Forward pass through branches, undo rotation on output predictions results
                    
            self.output_prob.append([
                None,
                F.grid_sample(self.graspnet(interm_percept_feat), flow_grid_after, mode='nearest', align_corners=True),
            ])

           
            return self.output_prob, None