import os
import numpy as np
import cv2
import torch
import carla

from collections import deque, namedtuple
from PIL import Image

from bird_view.models.model_supervised import Model_Segmentation_Traffic_Light_Supervised
from bird_view.models.model_RL import DQN, Orders
from bird_view.models.config import GlobalConfig

from team_code.base_agent import BaseAgent
# jxy: addition; (add display.py and fix RoutePlanner.py)
from team_code.display import HAS_DISPLAY, Saver, debug_display
# addition from team_code/map_agent.py
from carla_project.src.common import CONVERTER, COLOR
from carla_project.src.carla_env import draw_traffic_lights, get_nearby_lights


def get_entry_point():
    return 'AgentIAsRL'


class AgentIAsRL(BaseAgent):

    def setup(self, path_to_conf_file=None):
        super().setup(path_to_conf_file)
        return AgentSaver

        # jxy: add return AgentSaver and init_ads (setup keep 5 lines); rm save_path;
    def init_ads(self, path_to_conf_file):
        self.config = GlobalConfig()
        args = self.config

        path_to_folder_with_model = args.path_folder_model
        path_to_model_supervised = os.path.join(path_to_folder_with_model, "model_supervised/")
        path_model_supervised = None
        for file in os.listdir(path_to_model_supervised):
            if ".pth" in file:
                if path_model_supervised is not None:
                    raise ValueError(
                        "There is multiple model supervised in folder " +
                        path_to_model_supervised +
                        " you must keep only one!",
                    )
                path_model_supervised = os.path.join(path_to_model_supervised, file)
        if path_model_supervised is None:
            raise ValueError("We didn't find any model supervised in folder " +
                             path_to_model_supervised)

        # All this magic number should match the one used when training supervised...
        model_supervised = Model_Segmentation_Traffic_Light_Supervised(
            len(args.steps_image), len(args.steps_image), 1024, 6, 4, args.crop_sky
        )
        model_supervised.load_state_dict(
            torch.load(path_model_supervised, map_location=args.device)
        )
        model_supervised.to(device=args.device)

        self.encoder = model_supervised.encoder
        self.last_conv_downsample = model_supervised.last_conv_downsample

        self.action_space = (args.nb_action_throttle + 1) * args.nb_action_steering

        path_to_model_RL = os.path.join(path_to_folder_with_model, "model_RL")
        tab_model = []
        for file in os.listdir(path_to_model_RL):
            if ".pth" in file:
                tab_model.append(os.path.join(path_to_model_RL, file))

        if len(tab_model) == 0:
            raise ValueError("We didn't find any RL model in folder "+ path_to_model_RL)

        self.tab_RL_model = []
        for current_model in tab_model:

            current_RL_model = DQN(args, self.action_space).to(device=args.device)
            current_RL_model_dict = current_RL_model.state_dict()

            print("we load RL model ", current_model)
            checkpoint = torch.load(current_model)

            # 1. filter out unnecessary keys
            pretrained_dict = {
                k: v
                for k, v in checkpoint["model_state_dict"].items()
                if k in current_RL_model_dict
            }
            # 2. overwrite entries in the existing state dict
            current_RL_model_dict.update(pretrained_dict)
            # 3. load the new state dict
            current_RL_model.load_state_dict(current_RL_model_dict)
            self.tab_RL_model.append(current_RL_model)

        self.window = (
            max([abs(number) for number in args.steps_image]) + 1
        )  # Number of frames to concatenate
        self.RGB_image_buffer = deque([], maxlen=self.window)
        self.device = args.device

        self.state_buffer = deque([], maxlen=self.window)
        self.State = namedtuple("State", ("image", "speed", "order", "steering"))

        if args.crop_sky:
            blank_state = self.State(
                np.zeros(6144, dtype=np.float32), -1, -1, 0
            )  # RGB Image, color channet first for torch
        else:
            blank_state = self.State(np.zeros(8192, dtype=np.float32), -1, -1, 0)
        for _ in range(self.window):
            self.state_buffer.append(blank_state)
            if args.crop_sky:
                self.RGB_image_buffer.append(
                    np.zeros((3, args.front_camera_height - 120, args.front_camera_width))
                )
            else:
                self.RGB_image_buffer.append(
                    np.zeros((3, args.front_camera_height, args.front_camera_width))
                )

        self.last_steering = 0
        self.last_order = 0

        self.current_timestep = 0

    def destroy(self): # jxy mv before _init
        torch.cuda.empty_cache()

        super().destroy()

    @torch.no_grad()
    def act(self, state_buffer, RL_model):
        speeds = []
        order = state_buffer[-1].order
        steerings = []
        for step_image in self.config.steps_image:
            state = state_buffer[step_image + self.window - 1]
            speeds.append(state.speed)
            steerings.append(state.steering)
        images = torch.from_numpy(state_buffer[-1].image).to(self.device, dtype=torch.float32)
        speeds = torch.from_numpy(np.stack(speeds).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )
        steerings = torch.from_numpy(np.stack(steerings).astype(np.float32)).to(
            self.device, dtype=torch.float32
        )

        quantile_values, _ = RL_model(
            images.unsqueeze(0),
            speeds.unsqueeze(0),
            order,
            steerings.unsqueeze(0),
            self.config.num_quantile_samples,
        )
        return quantile_values.mean(0).argmax(0).item()

    # We had different mapping int/order in our training than in the CARLA benchmark,
    # so we need to remap orders
    def adapt_order(self, incoming_obs_command):
        if incoming_obs_command == 1:  # LEFT
            return Orders.Left.value
        if incoming_obs_command == 2:  # RIGHT
            return Orders.Right.value
        if incoming_obs_command == 3:  # STRAIGHT
            return Orders.Straight.value
        if incoming_obs_command == 4:  # FOLLOW_LANE
            return Orders.Follow_Lane.value
        # jxy: add refer to ~/CARLA_0.9.11/PythonAPI/carla/agents/navigation/local_planner.py
        if incoming_obs_command == 5:  # CHANGELANELEFT
            return Orders.Left.value
        if incoming_obs_command == 6:  # CHANGELANERIGHT
            return Orders.Right.value
        if incoming_obs_command == -1:  # VOID
            return Orders.Follow_Lane.value

    def sensors(self):
        result = [sn for sn in super().sensors() if sn['id'] not in ['rgb']]
        result.append({
                        'type': 'sensor.camera.rgb',
                        'x': 1.5, 'y': 0.0, 'z': 2.4,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': 288, 'height': 288, 'fov': 90,
                        'id': 'rgb'
                    })
        return result

    def tick(self, input_data):
        result = super().tick(input_data) # self.step += 1 in super().tick

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        far_node, next_cmd = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        result['target'] = target
        # jxy addition:
        result['far_command'] = next_cmd
        result['R_pos_from_head'] = R
        result['offset_pos'] = np.array([gps[0], gps[1]])
        # from team_code/map_agent.py:
        self._actors = self._world.get_actors()
        self._traffic_lights = get_nearby_lights(self._vehicle, self._actors.filter('*traffic_light*'))
        topdown = input_data['map'][1][:, :, 2]
        topdown = draw_traffic_lights(topdown, self._vehicle, self._traffic_lights)
        result['topdown'] = COLOR[CONVERTER[topdown]]
        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)
        self.current_timestep += 1
        self.step = self.current_timestep

        rgb = tick_data['rgb'] # tick_data['image']
        if self.config.crop_sky:
            rgb = np.array(rgb)[120:, :, :]
        else:
            rgb = np.array(rgb)
        # if self.config.render:
        #     bgr = rgb[:, :, ::-1]
        #     cv2.imshow("network input", bgr)
        #     cv2.waitKey(1)

        rgb = np.rollaxis(rgb, 2, 0)
        self.RGB_image_buffer.append(rgb)

        speed = tick_data['speed']

        order = self.adapt_order(int(tick_data['far_command'].value))
        if self.last_order != order:
            print("order = ", Orders(order).name)
            self.last_order = order

        np_array_RGB_input = np.concatenate(
            [
                self.RGB_image_buffer[indice_image + self.window - 1]
                for indice_image in self.config.steps_image
            ]
        )
        torch_tensor_input = (
            torch.from_numpy(np_array_RGB_input)
            .to(dtype=torch.float32, device=self.device)
            .div_(255)
            .unsqueeze(0)
        )

        current_encoding = self.encoder(torch_tensor_input)
        current_encoding = self.last_conv_downsample(current_encoding)
        current_encoding_np = current_encoding.cpu().numpy().flatten()

        current_state = self.State(current_encoding_np, speed, order, self.last_steering)
        self.state_buffer.append(current_state)

        tab_action = []

        for RL_model in self.tab_RL_model:
            current_action = self.act(self.state_buffer, RL_model)
            tab_action.append(current_action)

        steer = 0
        throttle = 0
        brake = 0

        for action in tab_action:

            steer += (
                (action % self.config.nb_action_steering) - int(self.config.nb_action_steering / 2)
            ) * (self.config.max_steering / int(self.config.nb_action_steering / 2))
            if action < int(self.config.nb_action_steering * self.config.nb_action_throttle):
                throttle += (int(action / self.config.nb_action_steering)) * (
                    self.config.max_throttle / (self.config.nb_action_throttle - 1)
                )
                brake += 0
            else:
                throttle += 0
                brake += 1.0

        steer = steer / len(tab_action)
        throttle = throttle / len(tab_action)
        if brake < len(tab_action) / 2:
            brake = 0
        else:
            brake = brake / len(tab_action)

        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        # control.manual_gear_shift = False
        self.last_steering = steer

        if HAS_DISPLAY: # jxy: change
            debug_display(tick_data, control.steer, control.throttle, control.brake, self.step)

        self.record_step(tick_data, control) # jxy: add
        return control

    # jxy: add record_step
    def record_step(self, tick_data, control, pred_waypoint=[]):
        # draw pred_waypoint
        if len(pred_waypoint):
            pred_waypoint[:,1] *= -1
            pred_waypoint = tick_data['R_pos_from_head'].dot(pred_waypoint.T).T
        self._command_planner.run_step2(pred_waypoint, is_gps=False, store=False) # metadata['wp_1'] relative to ego head (as y)
        # addition: from leaderboard/team_code/auto_pilot.py
        speed = tick_data['speed']
        self._recorder_tick(control) # jxy trjs
        ego_bbox = self.gather_info() # jxy metrics
        self._command_planner.run_step2(ego_bbox, is_gps=False, store=False)
        self._command_planner.show_route()

        if self.save_path is not None and self.step % self.record_every_n_step == 0:
            self.save(control.steer, control.throttle, control.brake, tick_data)


# jxy: mv save in AgentSaver & rm destroy
class AgentSaver(Saver):
    def __init__(self, path_to_conf_file, dict_, list_):
        self.config_path = path_to_conf_file

        # jxy: according to sensor
        self.rgb_list = ['rgb', 'rgb_left', 'rgb_right', 'topdown'] #  'rgb_rear', 'rgb_with_car', 
        self.add_img = [] # 'flow', 'out', 
        self.lidar_list = [] # 'lidar_0', 'lidar_1',
        self.dir_names = self.rgb_list + self.add_img + self.lidar_list #+ ['pid_metadata']

        super().__init__(dict_, list_)

    def run(self): # jxy: according to init_ads

        super().run()

    def _save(self, tick_data):
        # addition
        # save_action_based_measurements = tick_data['save_action_based_measurements']
        self.save_path = tick_data['save_path']
        if not (self.save_path / 'ADS_log.csv' ).exists():
            # addition: generate dir for every total_i
            self.save_path.mkdir(parents=True, exist_ok=True)
            for dir_name in self.dir_names:
                (self.save_path / dir_name).mkdir(parents=True, exist_ok=False)

            # according to self.save data_row_list
            title_row = ','.join(
                ['frame_id', 'far_command', 'speed', 'steering', 'throttle', 'brake',] + \
                self.dir_names
            )
            with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
                f_out.write(title_row+'\n')

        self.step = tick_data['frame']
        self.save(tick_data['steer'],tick_data['throttle'],tick_data['brake'], tick_data)

    # addition: modified from leaderboard/team_code/auto_pilot.py jxy: add data_row_list
    def save(self, steer, throttle, brake, tick_data):
        # frame = self.step // 10
        frame = self.step

        # 'gps' 'thetas'
        pos = tick_data['gps']
        speed = tick_data['speed']
        far_command = tick_data['far_command']
        data_row_list = [frame, far_command.name, speed, steer, throttle, brake,]

        # images
        for rgb_name in self.rgb_list:
            path_ = self.save_path / rgb_name / ('%04d.png' % frame)
            Image.fromarray(tick_data[rgb_name]).save(path_)
            data_row_list.append(str(path_))

        # collection
        data_row = ','.join([str(i) for i in data_row_list])
        with (self.save_path / 'ADS_log.csv' ).open("a") as f_out:
            f_out.write(data_row+'\n')


