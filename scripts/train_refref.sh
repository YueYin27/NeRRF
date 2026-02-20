# singularity shell --nv /opt/apps/containers/pytorch22.01-libx11-2.sif
# Loop through all the datasets in the directory
for bg_dir in /home/projects/RefRef/image_data/textured_sphere_scene/*; do
   echo "bg_dir: $bg_dir"
   for dataset_dir in $bg_dir/*; do
   echo "dataset_dir: $dataset_dir"

      # # skip the directory named "ball"
      # if [ $(basename $dataset_dir) == "ball" ]; then
      #       continue
      # fi

      echo "Training on dataset: $(basename $bg_dir) $(basename $dataset_dir)"
      echo "Stage 1"
      CUDA_LAUNCH_BLOCKING=1 \
      python train/train.py \
      -n $(basename $dataset_dir) \
      -c NeRRF_stage1_new.conf \
      -D $dataset_dir \
      --gpu_id=0 \
      --visual_path tet_visual \
      --stage 1 \
      --tet_scale 4.2 \
      --sphere_radius 1.5 \
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
      -c NeRRF_stage2_new.conf \
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
done
