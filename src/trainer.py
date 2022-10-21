import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import reactive_net, reinforcement_net, reinforcement_static_net
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Trainer(object):
    def __init__(self, robot, training_target, future_reward_discount, frames,
                 is_testing, load_snapshot, snapshot_file, force_cpu, load_static_snapshot, static_snapshot_file):
        self.robot = robot
        self.training_target = training_target


        if training_target!="static_grasp":
            self.xy_moving_timetable = np.load(os.path.join(os.getcwd(), "timetables", "xy_timetable.npy"))
            self.rotate_timetable = np.load(os.path.join(os.getcwd(), "timetables", "rotate_timetable.npy"))
            z_timetable = np.load(os.path.join(os.getcwd(), "timetables", "z_timetable.npy"))[1:]

            self.z_time_predictor = LinearRegression()
            self.z_time_predictor.fit(np.array([i+1 for i in range(len(z_timetable))]).reshape(-1, 1), z_timetable.reshape(-1, 1))



        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        self.static_model = reinforcement_static_net(self.use_cuda)
        self.model = reinforcement_net(self.use_cuda, frames)
        self.future_reward_discount = future_reward_discount

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        # Load pre-trained model
        if load_static_snapshot:
            self.static_model.load_state_dict(torch.load(static_snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (static_snapshot_file))
        
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))
            
        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()
            self.static_model = self.static_model.cuda()

        # Set model to training mode
        if self.training_target=="static_grasp":
            self.static_model.train()
            self.optimizer = torch.optim.SGD(self.static_model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        else:
            self.model.train()
            self.static_model.eval()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0
        # if training_target=='static_grasp':
        #     if static_snapshot_file:
        #         self.iteration = int(snapshot_file.split("/")[-1].split("-")[1][:6])
        # else:
        #     if snapshot_file:
        #         self.iteration = int(snapshot_file.split("/")[-1].split("-")[1][:6])


        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.use_heuristic_log = []
        self.is_exploit_log = []
        self.clearance_log = []



    # Pre-load execution info and RL variables
    def preload(self, transitions_directory):
        self.executed_action_log = np.load(os.path.join(transitions_directory, 'executed-action.log.npy'), allow_pickle=True)
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.load(os.path.join(transitions_directory, 'label-value.log.npy'), allow_pickle=True)
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()

        # self.label_grasp_value_log = np.load(os.path.join(transitions_directory, 'label-grasp-value.log.npy'), allow_pickle=True)
        # self.label_grasp_value_log = self.label_grasp_value_log[0:self.iteration]
        # self.label_grasp_value_log.shape = (self.iteration,1)
        # self.label_grasp_value_log = self.label_grasp_value_log.tolist()
        
        self.predicted_value_log = np.load(os.path.join(transitions_directory, 'predicted-value.log.npy'), allow_pickle=True)
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.load(os.path.join(transitions_directory, 'reward-value.log.npy'), allow_pickle=True)
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()
        self.use_heuristic_log = np.load(os.path.join(transitions_directory, 'use-heuristic.log.npy'), allow_pickle=True)
        self.use_heuristic_log = self.use_heuristic_log[0:self.iteration]
        self.use_heuristic_log.shape = (self.iteration,1)
        self.use_heuristic_log = self.use_heuristic_log.tolist()
        self.is_exploit_log = np.load(os.path.join(transitions_directory, 'is-exploit.log.npy'), allow_pickle=True)
        self.is_exploit_log = self.is_exploit_log[0:self.iteration]
        self.is_exploit_log.shape = (self.iteration,1)
        self.is_exploit_log = self.is_exploit_log.tolist()
        if os.path.exists(os.path.join(transitions_directory, 'clearance.log.npy')):
            self.clearance_log = np.load(os.path.join(transitions_directory, 'clearance.log.npy'), allow_pickle=True)
            self.clearance_log.shape = (self.clearance_log.shape[0],1)
            self.clearance_log = self.clearance_log.tolist()
        
    
    def get_analitical_velocity(self, prediction, depth_heightmap):
        best_pix_x = prediction[2]
        best_pix_y = prediction[1]
        z_coo = depth_heightmap[best_pix_y][best_pix_x] + self.robot.workspace_limits[2][0]
        z_coo = max(z_coo - 0.04, self.robot.workspace_limits[2][0] + 0.02)
        z_time = self.z_time_predictor.predict(np.array(z_coo).reshape(-1, 1)).item()
        t = self.xy_moving_timetable[tuple(prediction[1:])] + self.rotate_timetable[prediction[0]] + z_time -2

        a = 0.05
        x = 0.5

        v0 = self.robot.get_conveyor_velocity()
    
        base_velocity = min(a*t + v0 + np.sqrt((a*t+v0)**2 - (2*a*x+v0**2)), a*t + v0 - np.sqrt((a*t+v0)**2 - (2*a*x+v0**2)))
        if np.isnan(base_velocity):
            base_velocity = (a*t + v0)
        return base_velocity
        
    # Compute forward pass through model to compute affordances/Q
    def preprocessing_input_img(self, color_heightmap, depth_heightmap):
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32 
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = 0.01
        image_std = 0.03
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = depth_heightmap_2x.copy()
        input_depth_image = (input_depth_image - image_mean)/image_std

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)
        return input_color_data, input_depth_data, padding_width, color_heightmap_2x

    def forward(self, color_heightmaps, depth_heightmaps, is_volatile=False, specific_rotation=-1, nonlocal_variables=None, teacher_student=False):
        input_color_data = []
        input_depth_data = []
        for color_heightmap, depth_heightmap in zip(color_heightmaps, depth_heightmaps):
            color, depth, padding_width, color_heightmap_2x = self.preprocessing_input_img(color_heightmap, depth_heightmap)
            input_color_data.append(color)
            input_depth_data.append(depth)
        
        # Pass input data through model
        if self.training_target == "static_grasp":
            output_prob, state_feat = self.static_model.forward(input_color_data[0], input_depth_data[0], is_volatile, specific_rotation)
        else:
            #static_modelからgrasp位置を決める
            index = 0
            if teacher_student:
                index = -1
            with torch.no_grad():
                output_prob, _ = self.static_model.forward(input_color_data[index], input_depth_data[index], is_volatile, specific_rotation)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    # regulate_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    static_grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    # regulate_predictions = np.concatenate((regulate_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    static_grasp_predictions = np.concatenate((static_grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
            static_grasp_idx = np.unravel_index(np.argmax(static_grasp_predictions), static_grasp_predictions.shape)
            base_velocity = self.get_analitical_velocity(static_grasp_idx, depth_heightmap)            
            output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation, base_velocity)
        
        if self.training_target == "static_grasp":
            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    # regulate_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    # regulate_predictions = np.concatenate((regulate_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
            return None, grasp_predictions, state_feat

        else:
            # Return Q values (and remove extra padding)
            for rotate_idx in range(len(output_prob)):
                if rotate_idx == 0:
                    regulate_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                    grasp_predictions = output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]
                else:
                    regulate_predictions = np.concatenate((regulate_predictions, output_prob[rotate_idx][0].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)
                    grasp_predictions = np.concatenate((grasp_predictions, output_prob[rotate_idx][1].cpu().data.numpy()[:,0,int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2),int(padding_width/2):int(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return regulate_predictions, grasp_predictions, base_velocity, static_grasp_predictions


    def get_label_value(self, primitive_action, regulate_success, grasp_success, regulate_velocity, prev_grasp_predictions, next_color_heightmap, next_depth_heightmap, nonlocal_variables, multi_steps=False):
        touch_object = nonlocal_variables['touch_object']

        # Compute current reward
        current_reward = 0

        if grasp_success:
            if multi_steps:
                current_reward += sum([item["grasp_success"] for item in nonlocal_variables["multi_pix_ind"] if item['grasp_success']]) / len(nonlocal_variables["multi_pix_ind"])
            else:
                current_reward += 1
        elif touch_object:
            current_reward += touch_object//3
                # touch_object = 0
        if self.training_target == "static_grasp":
            next_regulate_predictions, next_grasp_predictions, next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True, nonlocal_variables=nonlocal_variables)
        else:
            next_regulate_predictions, next_grasp_predictions,_,  next_state_feat = self.forward(next_color_heightmap, next_depth_heightmap, is_volatile=True, nonlocal_variables=nonlocal_variables)
        future_reward = np.max(next_grasp_predictions)

        print(primitive_action, 'Current reward: %f' % (current_reward))
        print(primitive_action, 'Future reward: %f' % (future_reward))
        expected_reward = current_reward + self.future_reward_discount * future_reward
        print(primitive_action, 'Expected reward: %f + %f x %f = %f' % (current_reward, self.future_reward_discount, future_reward, expected_reward))
        print()
        return expected_reward, current_reward

    def multi_steps_backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, regulate_residual=None, nonlocal_variables=None):
        loss_value = 0 

        # Compute labels
        for rotate in range(self.model.num_rotations):
            _x = [item["x"] for item in nonlocal_variables["multi_pix_ind"] if item["rotate"]==rotate and item["grasp_success"]]
            if len(_x)==0:
                continue
            _y = [item["y"] for item in nonlocal_variables["multi_pix_ind"] if item["rotate"]==rotate and item["grasp_success"]]
            
            label = np.zeros((1,320,320))
            action_area = np.zeros((224,224))
            action_area[_y, _x] = 1

            # blur_kernel = np.ones((5,5),np.float32)/25
            # action_area = cv2.filter2D(action_area, -1, blur_kernel)
            tmp_label = np.zeros((224,224))
            if regulate_residual:
                tmp_label[action_area > 0] = regulate_residual
            else:
                tmp_label[action_area > 0] = label_value
            
            label[0,48:(320-48),48:(320-48)] = tmp_label

            # Compute label mask
            label_weights = np.zeros(label.shape)
            tmp_label_weights = np.zeros((224,224))
            tmp_label_weights[action_area > 0] = 1
            label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

            # Compute loss and backward pass
            self.optimizer.zero_grad()
                            
                # Do forward pass with specified rotation (to save gradients)
            if self.training_target == "static_grasp":
                regulate_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=rotate, nonlocal_variables=nonlocal_variables)
            else:
                regulate_predictions, grasp_predictions, _, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=rotate, nonlocal_variables=nonlocal_variables)
        
            if self.training_target == "static_grasp":
                output_prob = self.static_model.output_prob
            else:
                output_prob = self.model.output_prob


            if self.use_cuda:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()

            loss.backward()
            loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (rotate + self.model.num_rotations/2) % self.model.num_rotations

            
            
            if self.training_target == "static_grasp":
                regulate_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, nonlocal_variables=nonlocal_variables)
                output_prob = self.static_model.output_prob
            else:
                regulate_predictions, grasp_predictions, _, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, nonlocal_variables=nonlocal_variables)
                output_prob = self.model.output_prob

            if self.use_cuda:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            
            
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value += loss_value/2
            
            print('Training loss: %f' % (loss_value))
            self.optimizer.step()
            return loss_value

    # Compute labels and backpropagate
    def backprop(self, color_heightmap, depth_heightmap, primitive_action, best_pix_ind, label_value, regulate_residual=None, nonlocal_variables=None, teacher_student=False):
        # Compute labels
        label = np.zeros((1,320,320))
        action_area = np.zeros((224,224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1

        # blur_kernel = np.ones((5,5),np.float32)/25
        # action_area = cv2.filter2D(action_area, -1, blur_kernel)
        tmp_label = np.zeros((224,224))
        if regulate_residual:
            tmp_label[action_area > 0] = regulate_residual
        else:
            tmp_label[action_area > 0] = label_value
        
        label[0,48:(320-48),48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros(label.shape)
        tmp_label_weights = np.zeros((224,224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0,48:(320-48),48:(320-48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0
        if 'regulate' in primitive_action :
            # Do forward pass with specified rotation (to save gradients)
            if self.training_target=="static_grasp":
                regulate_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], nonlocal_variables=nonlocal_variables)                
            else:
                regulate_predictions, grasp_predictions, _, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], nonlocal_variables=nonlocal_variables)                
            

            if self.use_cuda:
                loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(self.model.output_prob[0][0].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()
            

        elif primitive_action == 'grasp':

            # Do forward pass with specified rotation (to save gradients)
            if self.training_target == "static_grasp":
                regulate_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], nonlocal_variables=nonlocal_variables)
            else:
                regulate_predictions, grasp_predictions, _, static_grasp_predictions = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0], nonlocal_variables=nonlocal_variables)
            if teacher_student:
                label[0,48:(320-48),48:(320-48)] = static_grasp_predictions[0]

            if self.training_target == "static_grasp":
                output_prob = self.static_model.output_prob
            else:
                output_prob = self.model.output_prob    
            if self.use_cuda:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)
            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            opposite_rotate_idx = (best_pix_ind[0] + self.model.num_rotations/2) % self.model.num_rotations

            
            
            if self.training_target == "static_grasp":
                regulate_predictions, grasp_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, nonlocal_variables=nonlocal_variables)
                output_prob = self.static_model.output_prob
            else:
                regulate_predictions, grasp_predictions, _, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=opposite_rotate_idx, nonlocal_variables=nonlocal_variables)
                output_prob = self.model.output_prob

            if self.use_cuda:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float().cuda())) * Variable(torch.from_numpy(label_weights).float().cuda(),requires_grad=False)
            else:
                loss = self.criterion(output_prob[0][1].view(1,320,320), Variable(torch.from_numpy(label).float())) * Variable(torch.from_numpy(label_weights).float(),requires_grad=False)

            loss = loss.sum()
            loss.backward()
            loss_value = loss.cpu().data.numpy()

            loss_value = loss_value/2
        else:
            raise NotImplementedError()

        print('Training loss: %f' % (loss_value))
        self.optimizer.step()
        return loss_value
    
    def min_max(self, x, axis=None):
        min, max = x.min(), x.max()
        return (x-min)/(max-min)

    # def visualization_for_find_grasp_point(self, predictions):
    #     predictions = torch.tensor(self.min_max(predictions))
    #     predictions = nn.MaxPool2d(kernel_size=kernel_size, padding=padding)(predictions).numpy()
        
    #     predictions = np.array([cv2.resize(p, (224, 224)) for p in predictions])
    #     predictions = np.where(predictions>0.8, 1, 0)

        
    def get_multi_targets_pix_ind(self, predictions_, best_pix_ind, explore_actions=None,kernel_size=(11, 11), stride=None, padding=5):
        predictions = torch.tensor(self.min_max(predictions_))
        predictions = nn.MaxPool2d(kernel_size=kernel_size, padding=padding)(predictions).numpy()
        
        predictions = np.array([cv2.resize(p, (224, 224)) for p in predictions])
        predictions = np.where(predictions>0.6, 1, 0)

        predictions = np.array([cv2.applyColorMap((prediction*255).astype(np.uint8), cv2.COLORMAP_JET) for prediction in predictions])
        
        
        # if explore_actions:
        #     explore = np.random.randint(0, 16)
        #     centers_x, centers_y = self.get_objects_center_of_gravity(predictions[explore])
        #     print(f'*explore: use {explore} rotation instead of {best_pix_ind[0]}')
        # else:
        centers_x, centers_y = self.get_objects_center_of_gravity(predictions[best_pix_ind[0]])
        
        grasp_points = []
        for x,y in zip(centers_x, centers_y):
            value = -np.inf

            for i, prediction in enumerate(predictions_):
                if value < prediction[(y, x)]:
                    value = prediction[(y, x)]
                    idx = i
            grasp_points.append({"rotate":idx, "x":x, "y":y, "grasp_success":None})
  
        return grasp_points
            
    def get_objects_center_of_gravity(self, img):
        ret_x, ret_y = [], []
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = np.where(gray==gray.min(), 255, 0).astype(np.uint8)

        controus, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in controus:
            m = cv2.moments(c)
            ret_x.append(int(m["m10"]/(m["m00"]+1e-6)))
            ret_y.append(int(m["m01"]/(m["m00"]+1e-6)))
        return ret_x, ret_y

    def get_multi_prediction_vis(self, predictions, color_heightmap, best_pix_ind, multi_pix_ind=None, type=None, kernel_size=(11, 11), stride=None, padding=5):
        predictions = torch.tensor(self.min_max(predictions))
        predictions = nn.MaxPool2d(kernel_size=kernel_size, padding=padding)(predictions).numpy()
        
        predictions = np.array([cv2.resize(p, (224, 224)) for p in predictions])
        predictions = np.where(predictions>0.6, 1, 0)
        # predictions = self.visualization_for_find_grasp_point(predictions)

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                if multi_pix_ind:
                    grasp_points_x = [item["x"] for item in multi_pix_ind if item["rotate"]==rotate_idx]
                    grasp_points_y = [item["y"] for item in multi_pix_ind if item["rotate"]==rotate_idx]

                if type=="conveyor":
                    prediction_vis = (prediction_vis).copy()
                elif type!=None:
                    prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                objects_center_x, objects_center_y = self.get_objects_center_of_gravity(prediction_vis)
                
                for x, y in zip(objects_center_x, objects_center_y):
                    # plt.plot(x, y, marker='.')
                    prediction_vis = cv2.circle(prediction_vis, (int(x), int(y)), 1, (0,255,0), 2)
                if multi_pix_ind:
                    for x,y in zip(grasp_points_x, grasp_points_y):
                        prediction_vis = cv2.circle(prediction_vis, (int(x), int(y)), 1, (255,255,255), 2)
                if type=="picture":
                    prediction_vis = color_heightmap.copy()
                if rotate_idx == best_pix_ind[0]:
                    # some grasp points are duplicated, so execute counts and visual points look different numbers.
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas

        
    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind, type=None):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                # prediction_vis[prediction_vis < 0] = 0 # assume probability
                # prediction_vis[prediction_vis > 1] = 1 # assume probability
                if type=="conveyor":
                    prediction_vis = (prediction_vis).copy()
                else:
                    prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if type=="picture":
                    prediction_vis = color_heightmap.copy()
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas



    def grasp_heuristic(self, depth_heightmap):

        num_rotations = 16

        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            valid_areas[np.logical_and(rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,-25], order=0) > 0.02, rotated_heightmap - ndimage.interpolation.shift(rotated_heightmap, [0,25], order=0) > 0.02)] = 1
            # valid_areas = np.multiply(valid_areas, rotated_heightmap)
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_grasp_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_grasp_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])

            if rotate_idx == 0:
                grasp_predictions = tmp_grasp_predictions
            else:
                grasp_predictions = np.concatenate((grasp_predictions, tmp_grasp_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
        return best_pix_ind


    


