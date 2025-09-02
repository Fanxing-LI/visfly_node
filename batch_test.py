trajs = ["D", "B", "8"]
vs = ["1.0","2.0","3.0"]
vs = ["0.5","1.5","2.5"] + vs

# generate sh
base = "python src/visfly/scripts/visfly_node.py --tr {} --v {}"

import os, sys
proj_path = os.path.dirname(os.path.abspath(__file__)).split("obj_track")[0] + "obj_track/"

# get current c
# create sh file
with open(proj_path+f"catkin_ws/src/visfly/run_visfly.sh", "w") as f:
    for traj in trajs:
        for v in vs:
            f.write(base.format(traj, v) + "\n")
            # sleep 10s
            f.write("sleep 10\n")
            
