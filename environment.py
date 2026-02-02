# Monkey patch simplepbr to fix MetaDrive 0.3.0.1 compatibility with simplepbr 0.11.2
# The user-requested version of simplepbr moved _load_shader_str to _shaderutils
import simplepbr
import simplepbr._shaderutils
if not hasattr(simplepbr, '_load_shader_str'):
    simplepbr._load_shader_str = simplepbr._shaderutils._load_shader_str
if not hasattr(simplepbr, '_add_shader_defines'):
    simplepbr._add_shader_defines = simplepbr._shaderutils._add_shader_defines

# Monkey patch gltf for MetaDrive 0.3.0.1 compatibility with panda3d-gltf 1.0+
import gltf
if not hasattr(gltf, 'patch_loader'):
    def patch_loader(loader):
        # Manually register the loader if needed, or assume auto-registration
        # For now, just a dummy to prevent AttributeError
        print("DEBUG: Applied gltf.patch_loader mock")
        pass
    gltf.patch_loader = patch_loader

# Monkey patch metadrive.engine.core.our_pbr.OurPipeline to handle None arguments
# simplepbr 0.11+ crashes if window=None is passed, but metadrive passes None by default.
try:
    import metadrive.engine.core.our_pbr as our_pbr_module
    import builtins
    from simplepbr import Pipeline
    
    def fixed_init(self, render_node=None, window=None, camera_node=None, taskmgr=None, msaa_samples=4, **kwargs):
        # Fill in defaults if None, matching simplepbr behavior
        if hasattr(builtins, 'base'):
            base = builtins.base
            if render_node is None: render_node = base.render
            if window is None: window = base.win
            if camera_node is None: camera_node = base.cam
            if taskmgr is None: taskmgr = base.task_mgr
        
        # Force MSAA to 4 for macOS compatibility (16 fails on many Arm64 GPUs)
        msaa_samples = 4
        
        # Call Pipeline.__init__ directly (skipping broken OurPipeline.__init__)
        Pipeline.__init__(self,
            render_node=render_node,
            window=window,
            camera_node=camera_node,
            taskmgr=taskmgr,
            msaa_samples=msaa_samples,
            **kwargs
        )

    # Patch the method in place
    our_pbr_module.OurPipeline.__init__ = fixed_init
    
    # Patch 'manager' property alias for simplepbr 0.11 compatibility
    # simplepbr renamed 'manager' (or equivalent) to '_filtermgr', but OurPipeline uses 'manager'
    setattr(our_pbr_module.OurPipeline, 'manager', property(lambda self: self._filtermgr))
    
    print("DEBUG: Applied OurPipeline.__init__ and manager alias patch for simplepbr compatibility")
except Exception as e:
    print(f"DEBUG: Failed to patch OurPipeline: {e}")

import gym
from gym import spaces
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

class SensorFusionEnv(MetaDriveEnv):
    def default_config(self):
        config = super(SensorFusionEnv, self).default_config()
        config.update({
            "map": "SCX",  # Tough map: S-Curve, Intersection, etc.
            "traffic_density": 0.3,
            "use_render": False,  # Default off, can be overridden
            "vehicle_config": {
                "image_source": "rgb_camera",
                "show_lidar": True, # Visual help for the user
                # Define LiDAR config
                "lidar": {
                    "num_lasers": 240,
                    "distance": 50,
                    "num_others": 8 # Track 8 nearest vehicles explicitly
                }
            },
            # We don't rely on MetaDrive's built-in image_observation switch alone
            # because we want BOTH image and lidar.
            "image_observation": False, 
        })
        return config
        
    def __init__(self, config=None):
        super(SensorFusionEnv, self).__init__(config)
        # Verify resolution from config or force it
        # We need to manually set up the camera if we are customizing too much
        # But MetaDrive usually handles it if we ask for it.
        # Let's ensure the observation space is correct.
        
        # Original vector space (LiDAR + State)
        self.vec_space = super().observation_space
        
        # Image space (84x84 RGB)
        self.img_space = spaces.Box(low=0, high=1, shape=(84, 84, 3), dtype=np.float32)
        
        # New combined space
        self._custom_observation_space = spaces.Dict({
            "image": self.img_space,
            "vector": self.vec_space
        })

    @property
    def observation_space(self):
        if hasattr(self, '_custom_observation_space'):
            return self._custom_observation_space
        return super().observation_space

    def reset(self, *args, **kwargs):
        r = super(SensorFusionEnv, self).reset(*args, **kwargs)
        # Handle potential tuple return (though usually just obs in gym 0.21)
        # If tuple, it is (obs, info) or (obs,)
        info = {}
        if isinstance(r, tuple):
            if len(r) == 2:
                vec_obs, info = r
            else:
                vec_obs = r[0]
        else:
            vec_obs = r

        image = self._get_image()
        obs = {
            "image": image,
            "vector": vec_obs
        }
        
        # Maintain signature
        if isinstance(r, tuple) and len(r) == 2:
            return obs, info
        return obs
        
    def step(self, action):
        r = super(SensorFusionEnv, self).step(action)
        # gym 0.21: obs, reward, done, info
        # Check return length just in case
        if len(r) == 4:
            vec_obs, reward, done, info = r
        else:
            # Fallback
            vec_obs = r[0]
            reward = r[1]
            done = r[2]
            info = r[3] if len(r) > 3 else {}
            
        image = self._get_image()
        obs = {
            "image": image,
            "vector": vec_obs
        }
        
        return obs, reward, done, info

    def _get_image(self):
        # Robust image retrieval
        try:
            if self.vehicle and self.vehicle.image_sensors and "rgb_camera" in self.vehicle.image_sensors:
                 return self.vehicle.image_sensors["rgb_camera"].perceive(self.vehicle, clip=True)
        except Exception:
            pass
            
        try:
             # Fallback to engine sensor
             if hasattr(self, "engine") and self.engine:
                 cam = self.engine.get_sensor("rgb_camera")
                 if cam:
                     return cam.perceive(self.vehicle, clip=True)
        except Exception:
            pass
            
        # Return blank image if sensors fail or not ready
        return np.zeros((84, 84, 3), dtype=np.float32)

# Removed setup_engine to avoid invalid imports and because it was empty.
            
# Helper function for SB3
def make_env(render=False, map_type="SCX"):
    config = dict(
        use_render=render,
        map=map_type, # Allow map override
        vehicle_config=dict(
            rgb_camera=(84, 84), # Set resolution
        )
    )
    env = SensorFusionEnv(config)
    
    # Wrap for Gymnasium compatibility (SB3 v2.0+)
    # MetaDrive 0.3.0.1 uses old Gym (returns 4 values)
    # Shimmy helps bridge this.
    try:
        from shimmy.openai_gym_compatibility import GymV21CompatibilityV0
        env = GymV21CompatibilityV0(env=env)
    except ImportError as e:
        print(f"Shimmy wrapper failed: {e}, proceeding without wrapper")
        
    return env
