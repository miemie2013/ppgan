

目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


cd ~/ppgan/custom_ops/gather
python setup.py install


cd custom_ops/gather
python setup.py install
cd ../../


解压数据集：
nvidia-smi
cd ~
cd data
cd data110820
unzip faces.zip
cd ~/ppgan


nvidia-smi
cd ~
cd data
cd data42681
unzip afhq.zip
cd ~/ppgan




看显存占用、GPU利用率
watch -n 0.1 nvidia-smi


训练模型（迁移学习）:
cd ~/ppgan
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams


nohup python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams > stylegan2ada.log 2>&1 &



单机双卡训练：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu

cd ~/ppgan
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --load styleganv2ada_512_afhqcat.pdparams




用从头训练 afhqcat ：
cd ~/ppgan
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml


nohup python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml > stylegan2ada.log 2>&1 &



------------------------ 恢复训练 ------------------------
cd ~/ppgan
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams


nohup python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams > stylegan2ada.log 2>&1 &



cd ~/ppgan
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --resume output_dir/stylegan_v2ada_256_custom_2_gpu-2022-03-18-17-02/iter_30000_checkpoint.pdparams




------------------------ 预测 ------------------------




------------------------ style-mixing ------------------------
cd ~/ppgan
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --style-mixing --load output_dir/stylegan_v2ada_256_custom-2022-04-20-10-08/iter_5000_checkpoint.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6


python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --style-mixing --load output_dir/stylegan_v2ada_256_custom-2022-04-20-10-08/iter_5000_checkpoint.pdparams --row_seeds 85,100 --col_seeds 55,821 --col_styles 0,1,2,3,4,5,6





------------------------ 计算指标 ------------------------
转换inceptionv3的权重：
cd convert_weights
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
python inception_convert_weights.py
cd ..



cd ~/ppgan
python tools/calc_metrics.py -c configs/stylegan_v2ada_512_afhqcat.yaml --load styleganv2ada_512_afhqcat.pdparams -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pdparams


论文中 afhqcat 的结果为3.55
实际可以跑得3.047967

用从头训练 afhqcat 的保存的模型：
cd ~/ppgan
python tools/calc_metrics.py -c configs/stylegan_v2ada_512_afhqcat.yaml --load output_dir/stylegan_v2ada_512_afhqcat-2022-03-18-17-02/iter_30000_checkpoint.pdparams -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pdparams






