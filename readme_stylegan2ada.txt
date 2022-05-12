

目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


cd ~/ppgan/custom_ops/gather
python setup.py install


cd custom_ops/gather
python setup.py install
cd ../../


如果报错 fatal error: cuda_runtime.h: 没有那个文件或目录

sudo vi ~/.bashrc

或者
sudo gedit ~/.bashrc

...
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${CUDA_HOME}/targets/x86_64-linux/include
export C_INCLUDE_PATH=$C_INCLUDE_PATH:${CUDA_HOME}/targets/x86_64-linux/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:${CUDA_HOME}/targets/x86_64-linux/lib
export C_INCLUDE_PATH=$C_INCLUDE_PATH:${CUDA_HOME}/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/targets/x86_64-linux/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${CUDA_HOME}/lib64

source ~/.bashrc

如果报错
cannot find -lcudart
collect2: error: ld returned 1 exit status
error: command '/usr/bin/g++' failed with exit code 1

find /usr/local/cuda-11.1 -name *cudart*


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


单机4卡训练：
cd ~/ppgan
CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml


恢复训练:
cd ~/ppgan
CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --resume output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams




nohup python -m paddle.distributed.launch --gpus 0,1,2,3 tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml > stylegan2ada.log 2>&1 &




------------------------ 恢复训练 ------------------------
cd ~/ppgan
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams


nohup python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams > stylegan2ada.log 2>&1 &



cd ~/ppgan
CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus 0,1 tools/main.py -c configs/stylegan_v2ada_256_custom_2_gpu.yaml --resume output_dir/stylegan_v2ada_256_custom_2_gpu-2022-03-18-17-02/iter_30000_checkpoint.pdparams




------------------------ 预测 ------------------------
cd ~/ppgan
CUDA_VISIBLE_DEVICES=0
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --evaluate-only --load output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams




------------------------ style-mixing ------------------------
cd ~/ppgan
CUDA_VISIBLE_DEVICES=0
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --style-mixing --load output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams --row_seeds 85,100 --col_seeds 55,821 --col_styles 0,1,2,3,4,5,6

python tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --style-mixing --load output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6




------------------------ 计算指标 ------------------------
转换inceptionv3的权重：
cd convert_weights
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
python inception_convert_weights.py
cd ..



cd ~/ppgan
python tools/calc_metrics.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --load output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pdparams


论文中 afhqcat 的结果为3.55
实际可以跑得3.047967

用从头训练 afhqcat 的保存的模型：
cd ~/ppgan
python tools/calc_metrics.py -c configs/stylegan_v2ada_512_afhqcat.yaml --load output_dir/stylegan_v2ada_512_afhqcat-2022-03-18-17-02/iter_30000_checkpoint.pdparams -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pdparams






