

目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


cd ~/ppgan/custom_ops/gather
python setup.py install


cd custom_ops/gather
python setup.py install
cd ../../



看显存占用、GPU利用率
watch -n 0.1 nvidia-smi


训练模型（迁移学习）:
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams


nohup python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams > stylegan2ada.log 2>&1 &



单机双卡训练：
cd ~/w*
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --load styleganv2ada_512_afhqcat.pdparams



------------------------ 恢复训练 ------------------------
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams


nohup python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams > stylegan2ada.log 2>&1 &



cd ~/w*
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --resume output_dir/stylegan_v2ada_256_custom_2_gpu-2022-03-18-17-02/iter_30000_checkpoint.pdparams







