python train.py --bs 5 \
    --gpu 0\
    --dataset nyu\
    --lr 0.000357\
    --wd 0.1\
    --div_factor 25\
    --final_div_factor 100\
    --epochs 10\
    --workers 11\
    --name UnetAdaptiveBins\
    --n_bins 128\
    --norm linear\
    --w_chamfer 0.0\
    --root .\
    --data_path ../dataset/nyu_depth_v2/sync/\
    --gt_path ../dataset/nyu_depth_v2/sync/\
    --filenames_file ./train_test_inputs/nyudepthv2_train_files_with_gt.txt\
    --input_height 416\
    --input_width 544\
    --min_depth 0.001\
    --max_depth 10\
    --do_random_rotate\
    --degree 2.5\
    --validate_every 500\
    --data_path_eval ../dataset/nyu_depth_v2/official_splits/test/\
    --gt_path_eval ../dataset/nyu_depth_v2/official_splits/test/\
    --filenames_file_eval ./train_test_inputs/nyudepthv2_test_files_with_gt.txt\
    --min_depth_eval 1e-3\
    --max_depth_eval 10\
    --eigen_crop