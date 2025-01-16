CUDA_VISIBLE_DEVICES=0 python inference.py \
--super_resolution \
--base_model_path "./checkpoint/MVD_weights/" \
--source_imgs_dir "./dataset/DTU_3views_save_warp-2/scan21_colmap/reference_images/" \
--warp_root_dir "./dataset/DTU_3views_save_warp-2/scan21_colmap/warp_images/" \
--output_dir "./output/scan21/" 
