for dataset_dir in /home/projects/RefRef/image_data/env_map_scene/multiple-non-convex/*; do
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
done
