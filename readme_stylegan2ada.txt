







训练模型（迁移学习）:
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams


单机双卡训练：
cd ~/w*
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --load styleganv2ada_512_afhqcat.pdparams



------------------------ 恢复训练 ------------------------
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams


cd ~/w*
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --resume output_dir/stylegan_v2ada_256_custom_2_gpu-2022-03-18-17-02/iter_30000_checkpoint.pdparams







