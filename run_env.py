from omni.isaac.kit import SimulationApp

app_config = {"headless": True}
simulation_app = SimulationApp(app_config)
import os
import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp
# import copy
from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric
from env.isaac_env_vs import env



def main():
    root_path = os.path.dirname(os.path.realpath(__file__))

    isaac_env = env(root_path=root_path, render=True, physics_dt=1 / 60.0)
    # isaac_env.model.test()
    while simulation_app.is_running():
        isaac_env.run()
        
    simulation_app.close()


if __name__ == "__main__":
    main()
