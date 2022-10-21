# tossingsub

TossingBot submodules used for industial field.

## Dependencies

- Ubuntu 20.04 
  - NVIDIA FeForce RTX 3090
  - GPU Driver 460.91.03
  - CUDA 10.1
  - cuDNN 7.6.5
- Universal Robots UR5e
- Robotiq 2F-140

## Requirements

- Python 3.8.0
  - gym >= 0.21.0
  - numpy >= 1.16.0
  - sympy >= 1.9
  - pyglet >= 1.5.21
  - Pillow >= 8.4.0
  - pybullet >= 3.2.0
  - matplotlib >= 3.5.1
  - pickle-mixin >= 1.0.2
  - seaborn >= 0.11.2
  - pandas >= 1.4.0
  - torch == 1.8.1
  - torchvision == 0.9.1
  - torchaudio == 0.8.1
  - python3-tk

## Installation

    $ git clone git@github.com:takuya-ki/tossingsub.git; cd tossingsub
    $ pip install -r requirements.txt torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
    $ apt-get install python3-tk

## Usage

### Check models

    $ python src/disp_env.py
    $ python src/disp_robot.py --gripper_type EEE
    $ python src/disp_object.py --obj XXX (--mode check_coordinate)
    $ python src/visualize_networks.py

### Run grasp-and-X task

    $ python src/grasptoss.py --obj bottle --frame 600mm --box_dist -0.60 --transport_vel 60.0 --release_width 0.0372 --throw_vel 3.000 --renders --sim_view back (--moving_obj, --measure_time)
    $ python src/grasptoss.py --obj animal --frame 450mm --box_dist -1.00 --transport_vel 60.0 --release_width 0.032 --throw_vel 2.800 --renders --sim_view back (--moving_obj, --measure_time)
    $ python src/grasptoss.py --obj choco --frame 766mm --box_dist -0.65 --transport_vel 60.0 --release_width 0.034 --throw_vel 2.700 --renders --sim_view back (--moving_obj, --measure_time)

    $ python src/graspplace.py --gripper_type 2f140 --obj bottle --frame 766mm --box_dist -0.05 --transport_vel 60.0 --renders --sim_view back (--moving_obj, --measure_time)
    $ python src/graspplace.py --gripper_type softmatics --obj bottle --frame 600mm --box_dist -0.048 --transport_vel 25.0 --renders --sim_view back (--moving_obj, --measure_time)

  - Check the input options as `python src/graspX.py -h` before executing the command above.
  - Specify the gripper type. Default is set to "2f140".
  - Specify target object from tossingsub/model/object/YYY.urdf. Default is set to "bottle".
  - Specify tossing target frame from tossingsub/env/frame/ZZZmm.urdf. Default is set to "450mm".
  - Specify --box_dist [m] to change the distance from robot base to target box. Default is set to "60.0".
  - The arm's trasporting velocity of the joint motions will be set to a float value [deg/s].
  - Finger openning widths at throwing start and at throwing end are set between a float value 0.000 ~ 0.140 [m] according to the target object size.
  - The arm's target throwing velocity will be set to a float value between 0.000 ~ 2.620 [m/s] (0.000 ~ 191.0 [deg/s] (3.333 [rad/s])).
  - Option --renders is used to enable rendering in simulation.
  - Option --sim_view is used to change the view point of simulation. Please refer to the help of options.  
  - Option --measure_time is used to measure the operations times.

##### Evaluate tossing results

    $ python src/evaluate_grasptoss.py --obj bottle --frame 450mm --box_dist -0.60 --transport_vel 60.0 --release_width 0.045 --throw_vel 2.620 --renders --sim_view back --num_step 100

  - Check the input options as `python src/evaluate_grasptoss.py -h` before executing the command above.
  - Specify tossing target object from tossingsub/model/object/YYY.urdf. Default is set to "bottle".
  - Specify tossing target frame from tossingsub/env/frame/ZZZmm.urdf. Default is set to "450mm".
  - Specify --box_dist [m] to change the distance from robot base to target box. Default is set to "-0.52".
  - The arm's trasporting velocity of the joint motions will be set to a float value [deg/s]. Default is set to "60.0".
  - Finger openning widths at throwing start and at throwing end are set between a float value 0.000 ~ 0.140 [m] according to the target object size.
  - The arm's target throwing velocity will be set to a float value between 0.000 ~ 2.620 [m/s] (0.000 ~ 191.0 [deg/s] (3.333 [rad/s])).
  - Option --renders is used to enable rendering in simulation.
  - Option --sim_view is used to change the view point of simulation. Please refer to the help of options.  
  - Option --measure_time is used to measure the operations times.

### Train with Q-learning, DQN, DDQN, or AC

    $ python src/train_XXX_ur5etosser.py --obj YYY --frame ZZZmm --box_dist -0.52 --transport_vel 60.0 --reward_func success-contact --weightpath weight/YYY/XXX/reward_func/filename --renders --sim_view back

  - Replace XXX with q, dqn, ddqn, or ac for q-leaning, dqn, ddqn, or actor-critic respectively.
  - Check the input options as `python src/train_XXX_ur5etosser.py -h` before executing the command above.
  - Specify tossing target object from tossingsub/object/YYY.urdf. Default is set to "bottle".
  - Specify tossing target frame from tossingsub/env/frame/ZZZmm.urdf. Default is set to "450mm".
  - Specify --box_dist [m] to change the distance from robot base to target box. Default is set to "-0.52".
  - The arm's trasporting velocity of the joint motions will be set to a float value [deg/s]. Default is set to "60.0".
  - Specify one reward function of four reward functions prepared. Default is set to "success-contact".
  - To use pre-trained weights, specify a weight file path for the specified model. Default is set to `None` which means to start training from scratch.
  - Option --renders is used to enable rendering in simulation.
  - Option --sim_view is used to change the view point of simulation. Please refer to the help of options.  

##### Check learning results

    $ python src/plot_statistics_one_object.py                          # to evaluate learning results  
    $ python -m pickle output/train_results/{method}/{object}_XXX.pkl   # to show the results  

##### Run a tossing motion infered with a trained model

    $ python src/run_trained_XXX_model.py --obj YYY --frame ZZZmm --box_dist -0.52 --transport_vel 60.0 --weightpath weight/YYY/XXX/reward_func/filename --renders --sim_view back

  - Replace XXX with q, dqn, ddqn, or ac for q-leaning, dqn, ddqn, or actor-critic respectively.
  - Specify tossing target object from tossingsub/object/YYY.urdf. Default is set to "bottle".
  - Specify tossing target frame from tossingsub/env/frame/ZZZmm.urdf. Default is set to "450mm".
  - Specify --box_dist [m] to change the distance from robot base to target box. Default is set to "-0.52".
  - The arm's trasporting velocity of the joint motions will be set to a float value [deg/s]. Default is set to "60.0".
  - Specify a weight file path for the specified model.
  - Option --renders is used to enable rendering in simulation.
  - Option --sim_view is used to change the view point of simulation. Please refer to the help of options.  

## Author / Contributor

[Takuya Kiyokawa](https://takuya-ki.github.io/)  
