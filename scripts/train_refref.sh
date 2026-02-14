# Loop through all the datasets in the directory
for dataset_dir in /home/projects/RefRef/image_data/env_map_scene/multiple-non-convex/*; do

    #  # skip the directory if it's in the list ("ball_hdr" "ball_coloured_hdr")
    #  if [ "$(basename $dataset_dir)" = "ball_hdr" ] || [ "$(basename $dataset_dir)" = "ball_coloured_hdr" ]; then
    #      continue
    #  fi

     echo "Training on dataset: $(basename $dataset_dir)"
     echo "Stage 1"
     CUDA_LAUNCH_BLOCKING=1 \
     python train/train.py \
     -n $(basename $dataset_dir) \
     -c NeRRF_stage1.conf \
     -D $dataset_dir \
     --gpu_id=0 \
     --visual_path tet_visual \
     --stage 1 \
     --tet_scale 4.2 \
     --sphere_radius 1.0 \
     --enable_refr \
     --enable_refl \
     --ior 1.5 \
     --use_sdf \
     --use_wandb \
     --wandb_project "NeRRF"

     echo "Stage 2"
     CUDA_LAUNCH_BLOCKING=1 \
     python train/train.py \
     -n $(basename $dataset_dir) \
     -c NeRRF_stage2.conf \
     -D $dataset_dir \
     --gpu_id=0 \
     --visual_path tet_visual \
     --stage 2 \
     --tet_scale 4.2 \
     --sphere_radius 2.0 \
     --enable_refr \
     --enable_refl \
     --ior 1.5 \
     --use_sdf \
     --use_wandb \
     --wandb_project "NeRRF"
done
