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

      # # skip the directory is if it doesn't start with "dfgjlstw"
      # case $(basename $dataset_dir) in
      #   d*|f*|g*|j*|l*|s*|t*|w*)
      #     ;;
      #   *)
      #     continue
      #     ;;
      # esac

      echo "Processing on dataset: $(basename $bg_dir) $(basename $dataset_dir)"
      
      # if the checkpoint file doesn't exist, run stage 1

      if [ ! -f "checkpoints/$(basename $dataset_dir)/1/_iter" ]; then
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
      fi

      # if the checkpoint file doesn't exist, run stage 2
      if [ ! -f "checkpoints/$(basename $dataset_dir)/2/net" ]; then
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
         --wandb_project "NeRRF" \
         --resume
      fi

      # if outputs directory doesn't exist, run evaluation
      if [ ! -f "outputs/$(basename $dataset_dir)/rgb_images/r_99.png" ]; then
         echo "Evaluating on dataset: $(basename $dataset_dir)"
         CUDA_LAUNCH_BLOCKING=1 \
         python eval/eval_approx.py \
         -n $(basename $dataset_dir) \
         -c NeRRF_stage2.conf \
         -D $dataset_dir \
         --gpu_id=0 \
         --stage 2 \
         --tet_scale 4.2 \
         --sphere_radius 2.0 \
         --enable_refr \
         --enable_refl \
         --ior 1.5 \
         --use_sdf
      fi
   done
done
