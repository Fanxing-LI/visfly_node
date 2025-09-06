trajs = ["D", "B", "8"]
vs = ["1.0","2.0","3.0"]
vs = ["0.5","1.0","1.5","2.0","2.5","3.0"]

# generate sh
base = "python exps/vary_v/run.py -tr {} --v {} "

policys = [
    # "SHAC_NoCaliHeadV_Pos_Dis1.5_spd3.4_lessNoise_1.zip",
    # "SHAC_NoCaliHeadV_Dis4.5_6.zip",
    # "SHAC_NoCaliHeadV_Pos_Dis3.0_spd3.4_lessNoise_2.zip",
    # "PPO_NoRand_1.zip"
    "SHAC_deploy_5.zip"
]

import os, sys
proj_path = os.path.dirname(os.path.abspath(__file__)).split("obj_track")[0] + "obj_track/"
base = "python exps/vary_v/run.py --train 0 --velocity {} --traj {} --algorithm SHAC --weight {}"
# get current c
# create sh file
with open(proj_path+f"exps/vary_v/batch_run.sh", "w") as f:
    for policy_i, policy in enumerate(policys):
        for traj_i, traj in enumerate(trajs):
            # if policy_i == 2 and traj_i == 2:
            #     continue
            for v in vs:
                f.write(base.format(v, traj, policy) + "\n")
                # sleep 10s
                f.write("sleep 10\n")
