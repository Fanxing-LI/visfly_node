import numpy as np
import torch as th
import copy
import cv2
import os
import sys
from typing import Optional, List
from VisFly.utils.policies.td_policies import obs_as_tensor


class SaveNode:
    def __init__(self, path, attrs, env):
        """
        Initialize SaveNode for data collection and saving
        
        Args:
            path: Save path for data
            attrs: List of attributes to track (e.g., ['state', 'obs', 'action', 'reward'])
            env: Environment instance for extracting observation structure
        """
            
        self.path = path
        self.attrs = attrs
        
        # Initialize storage lists for each attribute
        for attr in attrs:
            setattr(self, attr + "_all", [])
            
        # Special tracking lists
        self.render_image_all = []
        self.center_all = []
        self.eq_r = []  # episode rewards
        self.eq_l = []  # episode lengths
        
        # Initialize image names from environment observation space
        obs = env._observations
        self._img_names = [name for name in obs.keys() if (("color" in name) or ("depth" in name) or ("semantic" in name))]
        self.max_semantic_id = 1e-6
        self.FPS = int(1/env.envs.dynamics.dt) 
        
        # Store initial observation
        self.stack(env)
        
        # Episode tracking
        self.agent_index = [i for i in range(env.num_envs)]
        
        print(f"SaveNode initialized, tracking {len(self.attrs)} attributes")
        print(f"Image types found: {self._img_names}")

    def stack(self, env):
        """
        Stack and process all environment data for current step
        
        Args:
            env: Environment instance
        """
        obs = env._observations
        try:
            # Update max semantic ID for proper video encoding
            if obs is not None and self._img_names:
                for name in self._img_names:
                    if "semantic" in name and name in obs:
                        semantic_max = obs[name].max().item()
                        self.max_semantic_id = max(self.max_semantic_id, semantic_max)
            
            # Store observations
            if obs is not None and 'obs' in self.attrs:
                obs_copy = copy.deepcopy(obs)
                # Add target information to observations
                obs_copy["target"] = copy.deepcopy(env.envs.dynamic_object_position)
                self.obs_all.append(obs_copy)
            
            # Store all tracked attributes
            for attr in self.attrs:
                if attr != 'obs':  # obs handled separately above
                    attr_list = getattr(self, attr + "_all")
                    if attr == 'target_dis':
                        target_dis = (env.target - env.position).norm(dim=1)
                        attr_list.append(target_dis)
                    else:
                        data = getattr(env, attr)
                        attr_list.append(data.clone())
            
            # Handle render images
            if env.visual:
                render_kwargs = {}
                imgs = env.render(**render_kwargs)
                
                # Add sub-video (color observation) to the main render image
                if obs is not None and self._img_names and len(self._img_names) > 0 and False:
                    obs_tensor = obs_as_tensor(obs, device="cpu")
                    edge = 0.01
                    shape_img = imgs[0].shape[:2]
                    edge_int = int(min(shape_img) * edge)
                    
                    # Use color observation as sub-image
                    sub_image = (obs_tensor["color"]).to(th.uint8).cpu().numpy()  # (N, C, H, W)
                    sub_image = np.transpose(sub_image, (0, 2, 3, 1))  # (N, H, W, C)
                    
                    replace_dim = (shape_img[0], shape_img[1] - sub_image.shape[1] - edge_int)
                    for i in range(len(obs_tensor["depth"])):
                        sub_image_shape = obs_tensor["depth"][i].shape[1:3]
                        replace_dim = (
                            replace_dim[0] - sub_image_shape[1] - edge_int,
                            replace_dim[1]
                        )
                        imgs[0][replace_dim[0]:(replace_dim[0] + sub_image_shape[0]),
                                replace_dim[1]:(replace_dim[1] + sub_image_shape[1]), :] = \
                            cv2.cvtColor(sub_image[i], cv2.COLOR_RGB2RGBA)
                
                render_image = cv2.cvtColor(imgs[0], cv2.COLOR_RGBA2RGB)
                self.render_image_all.append(render_image)
            
            
        except Exception as e:
            print(f"Error in SaveNode.stack(): {e}")
            import traceback
            traceback.print_exc()

    def save(self, path=None, remove_images=True):
        """
        Save collected data to file, matching the format used in exps/vary_v/run.py
        
        Args:
            path: Custom save path (uses self.path if None)
            remove_images: Whether to remove image data from saved observations
        """
        save_path = path or self.path
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        # Save data in .pth format
        if save_path.endswith('.pth'):
            pth_path = save_path
        else:
            pth_path = save_path + '.pth'
            
        # Save video if render images are available
        # Create video folder with same name as data file (without .pth extension)
        if self.render_image_all:
            video_folder_path = pth_path.replace('.pth', '') if pth_path.endswith('.pth') else pth_path
            self.save_video(video_folder_path)
                
        # Remove image data to save space if requested
        if remove_images:
            print("Removing image data from observations to save space...")
            for obs in self.obs_all:
                for key in list(obs.keys()):
                    if "color" in key or "depth" in key or "semantic" in key:
                        del obs[key]
        
        # Prepare data dictionary matching exps/vary_v/run.py format
        save_data = {
            "state_all": getattr(self, 'state_all', []),
            "obs_all": getattr(self, 'obs_all', []),
            "t": getattr(self, 't_all', []),
            "target_all": getattr(self, 'target_all', []),
            "collision_all": getattr(self, 'collision_all', []),
            "reward_all": getattr(self, 'reward_all', []),
            "action_all": getattr(self, 'action_all', []),
            "target_dis_all": getattr(self, 'target_dis_all', []),
            "center_all": getattr(self, 'box_center_all'),
            # "info_all": getattr(self, 'info_all', [])  # Commented out like in original
        }
        
        th.save(save_data, pth_path)
                
        print("======================================================================")
        print(f"Test results saved to {pth_path}")
        if self.eq_r:
            print(f"Episode statistics: {len(self.eq_r)} episodes completed")
            print(f"Mean reward: {th.tensor(self.eq_r).mean().item():.4f}")
            print(f"Mean length: {th.tensor(self.eq_l).mean().item():.2f}")
        print("======================================================================")
        
        return save_data

    def save_video(self, base_path, is_sub_video=True):
        """
        Save render images as video and sub-videos for observations, following TestBase pattern
        
        Args:
            base_path: Base path for creating video folder and saving video
            is_sub_video: Whether to save sub-videos for observations (depth, color, semantic)
        """
        if not self.render_image_all:
            print("No render images to save as video")
            return
        
        # Create folder for video (using base_path as folder name)
        video_folder = base_path
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        
        height, width, layers = self.render_image_all[0].shape
        
        # Main render video (following TestBase naming: video.mp4)
        video_path = f"{video_folder}/video.mp4"
        
        # Calculate FPS from stored time data
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = self.FPS
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Prepare sub-videos for observations
        path_obs = []
        video_obs = []
        if is_sub_video and self._img_names and self.obs_all:
            for name in self._img_names:
                path_obs.append(f"{video_folder}/{name}.mp4")
                # Get dimensions from first observation
                obs_width = self.obs_all[0][name].shape[3] * self.obs_all[0][name].shape[0]  # width * batch_size
                obs_height = self.obs_all[0][name].shape[2]  # height
                video_obs.append(cv2.VideoWriter(path_obs[-1], fourcc, fps, (obs_width, obs_height)))
        
        # Write frames
        for index, image in enumerate(self.render_image_all):
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video.write(image_bgr)
            
            # Write sub-video frames
            if is_sub_video and self._img_names and index < len(self.obs_all):
                obs = self.obs_all[index]
                for i, name in enumerate(self._img_names):
                    if name in obs:
                        if "depth" in name:
                            max_depth = 10
                            img = np.clip(np.hstack(np.transpose(obs[name], (0, 2, 3, 1))), None, max_depth)
                            img = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255 / max_depth).astype(np.uint8)
                            video_obs[i].write(img)
                        elif "color" in name:
                            img = np.hstack(np.transpose(obs[name], (0, 2, 3, 1)))
                            img = img.astype(np.uint8)
                            # Convert RGB to BGR for OpenCV
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                            video_obs[i].write(img)
                        elif "semantic" in name:
                            max_id = self.max_semantic_id if self.max_semantic_id > 1e-6 else 255
                            img = np.hstack(np.transpose(obs[name], (0, 2, 3, 1)))
                            img = (cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255 / max_id).astype(np.uint8)
                            video_obs[i].write(img)
        
        # Release all video writers
        video.release()
        if is_sub_video and video_obs:
            for i in range(len(video_obs)):
                video_obs[i].release()
        
        print(f"video saved in {video_path}")
        if is_sub_video and path_obs:
            for obs_path in path_obs:
                print(f"sub-video saved in {obs_path}")

    def get_statistics(self):
        """
        Get basic statistics about collected data
        
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            'active_agents': len(self.agent_index),
            'completed_episodes': len(self.eq_r),
            'mean_episode_reward': th.tensor(self.eq_r).mean().item() if self.eq_r else 0,
            'mean_episode_length': th.tensor(self.eq_l).mean().item() if self.eq_l else 0,
        }
        
        # Add data sizes for each attribute
        for attr in self.attrs:
            attr_data = getattr(self, attr + "_all", [])
            stats[f'{attr}_count'] = len(attr_data)
        
        return stats

    def is_complete(self):
        """
        Check if all agents have completed their episodes
        
        Returns:
            bool: True if all agents are done
        """
        return len(self.agent_index) == 0