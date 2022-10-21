
cd ..
# OBJ_NUM=60
OBJ_NUM=10
TRAINING_TARGET=static_grasp

STATIC_SNAPSHOT_FILE=/home/nagata/sources/vrg/save_weights/static_grasp/snapshot-012200.reinforcement.pth

# python3  main.py --is_sim --push_rewards --experience_replay --explore_rate_decay --save_visualizations --obj_mesh_dir objects/blocks --num_obj $OBJ_NUM --method reinforcement --grasp_only
python3 -m pdb src/vg.py \
    --experience_replay\
    --explore_rate_decay\
    --save_visualizations \
    --num_obj ${OBJ_NUM} \
    --training_target $TRAINING_TARGET\
    --render\
    --sim_view front\
    # --debug\
    # --gripper_type softmatics

    # --load_static_snapshot\
    # --static_snapshot_file $STATIC_SNAPSHOT_FILE\
   #--grasp_only



### restart from snapshot
# SNAPSHOT=/home/nagata/sources/vrg_residual/logs/2022-07-09.03:08:18/models/snapshot-backup.reinforcement.pth
# python3 main.py --is_sim --experience_replay --explore_rate_decay --save_visualizations --obj_mesh_dir objects/blocks --num_obj ${OBJ_NUM} --method reinforcement --load_snapshot --snapshot_file ${SNAPSHOT}
