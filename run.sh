#!/bin/sh
unzip ppgan.zip

python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html
pip install -r requirements.txt

cd /custom_ops/gather
python setup.py install
cd ~

cd /root/paddlejob/workspace/train_data/datasets/data42681/
unzip afhq.zip
cd ~


# train
python -m paddle.distributed.launch --gpus 0,1,2,3 tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml

# resume train
# python -m paddle.distributed.launch --gpus 0,1,2,3 tools/main.py -c configs/stylegan_v2ada_512_afhqcat_4_gpu.yaml --resume output_dir/stylegan_v2ada_512_afhqcat_4_gpu-2022-05-12-11-43/iter_20_checkpoint.pdparams


