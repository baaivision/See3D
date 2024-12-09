CUDA_VISIBLE_DEVICES=0 python inference.py \
--super_resolution \
--single_view \
--base_model_path "./checkpoint/MVD_weights/" \
--source_imgs_dir "./dataset/co3d_1views_save/12_104_640/reference_images/" \
--warp_root_dir "./dataset/co3d_1views_save/12_104_640/warp_images/" \
--output_dir "./output/12_104_640/"