
modulated_conv2d
conv2d_resample
upfirdn2d
_conv2d_wrapper


wget https://paddlegan.bj.bcebos.com/InceptionV3.pdparams

======================== StyleGANv2_ADA ========================


# 转换原版仓库权重
python convert_weights/stylegan2ada_convert_weights.py -c configs/stylegan_v2ada_512_afhqcat.yaml -c_Gema G_ema_afhqcat.pth -c_G G_afhqcat.pth -c_D D_afhqcat.pth -oc styleganv2ada_512_afhqcat.pdparams

python convert_weights/stylegan2ada_convert_weights.py -c configs/stylegan_v2ada_1024_metfaces_1_gpu.yaml -c_Gema G_ema_metfaces.pth -c_G G_metfaces.pth -c_D D_metfaces.pth -oc styleganv2ada_1024_metfaces.pdparams

python convert_weights/stylegan2ada_convert_weights.py -c configs/stylegan_v2ada_1024_metfaces_1_gpu.yaml -c_Gema G_ema_ffhq.pth -c_G G_ffhq.pth -c_D D_ffhq.pth -oc styleganv2ada_1024_ffhq.pdparams

# 测试
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --evaluate-only --load styleganv2ada_512_afhqcat.pdparams

python tools/main.py -c configs/stylegan_v2ada_1024_metfaces_1_gpu.yaml --evaluate-only --load styleganv2ada_1024_metfaces.pdparams

python tools/main.py -c configs/stylegan_v2ada_1024_metfaces_1_gpu.yaml --evaluate-only --load styleganv2ada_1024_ffhq.pdparams




1.因为会报错 ValueError: (InvalidArgument) float16 can only be used when CUDNN or NPU is used
所以强制设置StyleGANv2ADA_SynthesisNetwork的use_fp16 = False
class StyleGANv2ADA_SynthesisNetwork(nn.Layer):
    def __init__(self,
    ):
        ...
        for res in self.block_resolutions:
            ...
            use_fp16 = False


F.grid_sample()没有实现二阶梯度
paddle.gather_nd()没有实现二阶梯度，所以暂时不能复现self.augment_pipe


# 安装依赖
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 解压AFHQ数据集
nvidia-smi
cd ~
cd data
cd data42681
unzip afhq.zip
cd ~/w*

nvidia-smi
cd ~
cd data
cd data130402
unzip dog_128.zip
cd ~/w*

动漫头像数据集：
nvidia-smi
cd ~
cd data
cd data110820
unzip faces.zip
cd ~/w*


复制预训练权重：
cd ~/w*
cp ../data/data128401/styleganv2ada_512_afhqcat.pdparams styleganv2ada_512_afhqcat.pdparams


对齐梯度：
1.(原版仓库也要设置)设置 ppgan/models/styleganv2ada_model.py 的 StyleGANv2ADAModel 的
    self.augment_pipe = None
    self.style_mixing_prob = -1.0
    self.align_grad = True
解除上面语句的注释即可。
以及，对下面所有的以if self.align_grad:开头的代码块解除注释
if self.align_grad:
    xxx

2.(原版仓库也要设置) ppgan/models/styleganv2ada_model.py，计算loss_Gpl那里，pl_noise使用全1而不是随机数：
            pl_noise = paddle.randn(gen_img.shape) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
            # pl_noise = paddle.ones(gen_img.shape) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
注释掉第一行代码，第二行代码解除注释。

3.(原版仓库也要设置)设置 ppgan/models/generators/generator_styleganv2ada.py 的 SynthesisLayer 的
    self.use_noise = False
解除上面语句的注释即可。
4.(原版仓库也要设置)设置 ppgan/models/generators/generator_styleganv2ada.py 的 StyleGANv2ADA_SynthesisNetwork 的
    use_fp16 = False
咩酱已经改好，这里不用修改。
5.(原版仓库也要设置)设置 ppgan/models/discriminators/discriminator_styleganv2ada.py 的 StyleGANv2ADA_Discriminator 的
    use_fp16 = False
咩酱已经改好，这里不用修改。

6.ppgan/models/base_model.py 里，优化器要换成SGD，在方法def setup_optimizers(self, lr_G, lr_D, cfg)里，注释掉
self.optimizers[opt_name] = build_optimizer(cfg_, lr, parameters)
下面的语句解除注释：
if opt_name == 'generator':
    self.optimizers[opt_name] = paddle.optimizer.Momentum(parameters=parameters, learning_rate=0.001, momentum=0.9)
elif opt_name == 'discriminator':
    self.optimizers[opt_name] = paddle.optimizer.Momentum(parameters=parameters, learning_rate=0.002, momentum=0.9)



export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu


# 转换权重方便对齐
python convert_weights/stylegan2ada_convert_weights.py -c configs/stylegan_v2ada_32_custom.yaml -c_G G_00.pth -c_Gema G_ema_00.pth -c_D D_00.pth -oc styleganv2ada_32_00.pdparams


python convert_weights/stylegan2ada_convert_weights.py -c configs/stylegan_v2ada_32_custom.yaml -c_G G_19.pth -c_Gema G_ema_19.pth -c_D D_19.pth -oc styleganv2ada_32_19.pdparams


# 转换权重方便对齐（简单网络版）
python convert_weights/stylegan2ada_convert_weights_simple.py -c configs/stylegan_v2ada_32_custom.yaml -c_G G_00.pth -c_Gema G_ema_00.pth -c_D D_00.pth -oc styleganv2ada_32_00.pdparams


python convert_weights/stylegan2ada_convert_weights_simple.py -c configs/stylegan_v2ada_32_custom.yaml -c_G G_19.pth -c_Gema G_ema_19.pth -c_D D_19.pth -oc styleganv2ada_32_19.pdparams




CUDA_VISIBLE_DEVICES=0
python tools/main.py -c configs/stylegan_v2ada_32_custom.yaml --load styleganv2ada_32_00.pdparams


CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch tools/main.py -c configs/stylegan_v2ada_32_custom.yaml --load styleganv2ada_32_00.pdparams



python diff_weights.py --cp1 styleganv2ada_32_19.pdparams --cp2 output_dir/stylegan_v2ada_32_custom-2022-04-15-10-04/iter_20_checkpoint.pdparams --d_value 0.0005









cd ~/w*
cd convert_weights
python test2_stylegan2ada_afhqcat32.py
python test2_stylegan2ada_afhqcat32_G19D19.py

cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_32_custom.yaml --load styleganv2ada_32_afhqcat.pdparams

cd ..
python diff_weights_with_pytorch.py



------------------------ 安装自定义op ------------------------
目前只有develop分支支持二阶导数：
python -m pip install paddlepaddle-gpu==0.0.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/gpu/develop.html

cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


cd ~/work/custom_ops/gather
python setup.py install



cd ~/work/test_grad
python test2_54_Discriminator_grad_paddle.py






------------------------ 训练模型 ------------------------
转换inceptionv3的权重：
cd convert_weights
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
python inception_convert_weights.py
cd ..



cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml


训练模型（迁移学习）:
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --load styleganv2ada_512_afhqcat.pdparams

cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --load styleganv2ada_512_afhqcat.pdparams

cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_128_custom.yaml --load styleganv2ada_512_afhqcat.pdparams

cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_32_custom.yaml --load styleganv2ada_512_afhqcat.pdparams



------------------------ 恢复训练 ------------------------
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --resume output_dir/stylegan_v2ada_512_afhqcat-2022-03-03-11-11/iter_20_checkpoint.pdparams

cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --resume output_dir/stylegan_v2ada_256_custom-2022-03-18-17-02/iter_30000_checkpoint.pdparams


测试模型:
cd ~/w*
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --evaluate-only --load output_dir/stylegan_v2ada_512_afhqcat-2022-02-16-18-18/epoch_4_checkpoint.pdparams


python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --evaluate-only --load styleganv2ada_512_afhqcat.pdparams


------------------------ 模型style-mixing ------------------------
python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --style-mixing --load styleganv2ada_512_afhqcat.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6

python tools/main.py -c configs/stylegan_v2ada_512_afhqcat.yaml --style-mixing --load styleganv2ada_512_afhqcat.pdparams --row_seeds 85 --col_seeds 55 --col_styles 0,1,2,3,4,5,6


python tools/main.py -c configs/stylegan_v2ada_128_custom.yaml --style-mixing --load output_dir/stylegan_v2ada_128_custom-2022-03-17-11-52/iter_9000_checkpoint.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6


python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --style-mixing --load iter_26000_checkpoint.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6


python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --style-mixing --load output_dir/stylegan_v2ada_256_custom-2022-03-18-15-34/iter_27000_checkpoint.pdparams --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6


如果提示显存不足，请减少row_seeds和col_seeds随机种子数量，比如：
python tools/main.py -c configs/stylegan_v2ada_256_custom.yaml --style-mixing --load iter_26000_checkpoint.pdparams --row_seeds 100,75,458 --col_seeds 55,821 --col_styles 0,1,2,3,4,5,6



------------------------ 计算指标 ------------------------
cd ~/w*
python tools/calc_metrics.py -c configs/stylegan_v2ada_512_afhqcat.yaml --load styleganv2ada_512_afhqcat.pdparams -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pdparams





======================== PastaGAN ========================


1.暂时没看
def normalize(self, upper_img, lower_img, upper_clothes_mask, lower_clothes_mask,
方法里做了什么

2.暂时设置PastaGANDiscriminator的use_fp16 = False
class PastaGANDiscriminator(nn.Layer):
            # use_fp16 = (res >= fp16_resolution)
            use_fp16 = False

3.计算损失时，噪声all_gen_z，先假设num_gpus = 1

4.先假设  self.augment_pipe = None


5.
x = y.add_(x)
这种表达式没有二阶梯度，需要改写为
x = y + x



创建虚拟环境
conda create -n pasta python=3.8

conda activate pasta
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu



export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=4




# 解压mpv3d数据集
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

nvidia-smi
cd ~
cd data
cd data125981
unzip MPV3D_starganv2.zip
cd ~/w*


训练模型（微调）:
cd ~/w*
python tools/main.py -c configs/pastagan_256_mpv3d.yaml --load G_ema_256.pdparams

nohup python tools/main.py -c configs/pastagan_256_mpv3d.yaml --load G_ema_256.pdparams > nohup.log 2>&1 &


单机双卡训练：
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py -c configs/pastagan_256_mpv3d.yaml --load G_ema_256.pdparams


可能报错：
You may need to install 'nccl2' from NVIDIA official website: https://developer.nvidia.com/nccl/nccl-downloadbefore

安装nccl2
https://blog.csdn.net/sinat_29957455/article/details/112507826

临时环境变量：
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu



从头训练：
cd ~/w*
python tools/main.py -c configs/pastagan_256_mpv3d.yaml




恢复训练:
cd ~/w*
python tools/main.py -c configs/pastagan_256_mpv3d.yaml --resume output_dir/pastagan_256_mpv3d-2022-02-10-14-32/epoch_4_checkpoint.pdparams


测试模型:
cd ~/w*
python tools/main.py -c configs/pastagan_256_mpv3d.yaml --evaluate-only --load output_dir/pastagan_256_mpv3d-2022-02-10-14-32/epoch_4_checkpoint.pdparams


python tools/main.py -c configs/pastagan_256_mpv3d_pretrained.yaml --evaluate-only --load G_ema_256.pdparams


======================== StarGANv2 ========================


# 解压mpv3d数据集
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

nvidia-smi
cd ~
cd data
cd data125981
unzip MPV3D_starganv2.zip
cd ~/w*


训练模型:
cd ~/w*
wget https://paddlegan.bj.bcebos.com/models/starganv2_afhq.pdparams
python -u tools/main.py -c configs/starganv2_mpv3d.yaml --load starganv2_afhq.pdparams


恢复训练:
cd ~/w*
python -u tools/main.py -c configs/starganv2_mpv3d.yaml --resume output_dir/starganv2_mpv3d-2022-01-20-11-33/epoch_5_checkpoint.pdparams


测试模型:
cd ~/w*
python tools/main.py -c configs/starganv2_mpv3d.yaml --evaluate-only --load output_dir/starganv2_mpv3d-2022-01-19-16-13/epoch_5_checkpoint.pdparams


python tools/main.py -c configs/starganv2_mpv3d.yaml --evaluate-only --load epoch_5_checkpoint.pdparams





# 解压AFHQ数据集
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

nvidia-smi
cd ~
cd data
cd data42681
unzip afhq.zip
cd ~/w*




示例以AFHQ数据集为例。如果您想使用CelebAHQ数据集，可以在换一下配置文件。

训练模型:
cd ~/w*
python -u tools/main.py -c configs/starganv2_afhq.yaml

测试模型:
cd ~/w*
python tools/main.py -c configs/starganv2_afhq.yaml --evaluate-only --load starganv2_afhq.pdparams




训练模型:
cd ~/w*
python -u tools/main.py -c configs/starganv2_celeba_hq.yaml

测试模型:
cd ~/w*
python tools/main.py -c configs/starganv2_celeba_hq.yaml --evaluate-only --load ${PATH_OF_WEIGHT}








======================== StyleGANv2 ========================
用户使用如下命令中进行生成，可通过替换seed的值或去掉seed生成不同的结果：

```
cd applications/
python -u tools/styleganv2.py \
       --output_path <替换为生成图片存放的文件夹> \
       --weight_path <替换为你的预训练模型路径> \
       --model_type ffhq-config-f \
       --seed 233 \
       --size 1024 \
       --style_dim 512 \
       --n_mlp 8 \
       --channel_multiplier 2 \
       --n_row 3 \
       --n_col 5 \
       --cpu
```

python -u tools/styleganv2.py --output_path StyleGANv2_outputs --weight_path aaaa --model_type ffhq-config-f --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5 --cpu



python -u tools/styleganv2.py --output_path StyleGANv2_outputs --model_type ffhq-config-f --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5


python -u tools/styleganv2.py --output_path StyleGANv2_outputs --model_type ffhq-config-f --seed 233 --size 256 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --n_row 3 --n_col 5









# 解压COCO2017数据集
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

nvidia-smi
cd ~
cd data
cd data123985
tar -xvf images256x256.tar
cd ~/w*




打包代码（文件）
cd ~/w*
rm -f aaa1.zip
zip -r aaa1.zip ./applications
rm -f aaa2.zip
zip -r aaa2.zip ./data
rm -f aaa3.zip
zip -r aaa3.zip ./test
rm -f aaa4.zip
zip -r aaa4.zip ./test_tipc
rm -f aaa5.zip
zip -r aaa5.zip ./ppgan
rm -f aaa6.zip
zip -r aaa6.zip ./tools
rm -f aaa7.zip
zip -r aaa7.zip ./configs
rm -f aaa8.zip
zip -r aaa8.zip *.py
rm -f aaa9.zip
zip -r aaa9.zip *.txt



删除压缩包
cd ~/w*
rm -f aaa1.zip
rm -f aaa2.zip
rm -f aaa3.zip
rm -f aaa4.zip
rm -f aaa5.zip
rm -f aaa6.zip
rm -f aaa7.zip
rm -f aaa8.zip
rm -f aaa9.zip





启动训练
先修改配置文件的dataroot
dataroot: ../data/ffhq/images256x256/
dataroot: data/ffhq/images256x256/
dataroot: ./../data/data123985/images256x256/
和

占用6G显存
    num_workers: 1
    batch_size: 1

占用10.5G显存(默认)
    num_workers: 3
    batch_size: 3

占用21.5G显存
    num_workers: 8
    batch_size: 8

占用29G显存
    num_workers: 12
    batch_size: 12

修改
total_iters: 800000
为
total_iters: 200000


修改
lr_scheduler:
  learning_rate: 0.002
  milestones: [600000]
为
lr_scheduler:
  learning_rate: 0.008
  milestones: [150000]



训练模型:
cd ~/w*
wget https://paddlegan.bj.bcebos.com/InceptionV3.pdparams
python tools/main.py -c configs/stylegan_v2_256_ffhq.yaml


单机双卡训练：
CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch tools/main.py -c configs/stylegan_v2_256_ffhq.yaml


# 恢复训练
cd ~/w*
python tools/main.py -c configs/stylegan_v2_256_ffhq.yaml --resume output_dir/stylegan_v2_256_ffhq-2021-12-31-14-51/iter_3000_checkpoint.pdparams




训练结束后，需要使用 ``tools/extract_weight.py`` 来提取对应的权重给``applications/tools/styleganv2.py``来进行推理.
python tools/extract_weight.py output_dir/stylegan_v2_256_ffhq-2021-12-31-14-51/iter_3000_weight.pdparams --net-name gen_ema --output my_styleganv2.pdparams

(调试)
python tools/extract_weight.py ../output_dir/stylegan_v2_256_ffhq-2021-12-31-14-51/iter_3000_weight.pdparams --net-name gen_ema --output ../my_styleganv2.pdparams



python tools/styleganv2.py --output_path stylegan01 --weight_path my_styleganv2.pdparams --size 256


(调试)
python tools/styleganv2.py --output_path ../stylegan01 --weight_path ../my_styleganv2.pdparams --size 256



======================== 换性别 ========================
cd ~/w*
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install cmake -i https://pypi.tuna.tsinghua.edu.cn/simple


pip install dlib -i https://pypi.tuna.tsinghua.edu.cn/simple



cd applications
python -u tools/pixel2style2pixel.py --input_image 'D://PycharmProjects/PaddleGAN-develop/yingbb2.png' --output_path   'D://PycharmProjects/PaddleGAN-develop/psp_output/yingbb2' --model_type ffhq-inversion --seed 233 --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2

python -u tools/styleganv2editing.py --latent 'D://PycharmProjects/PaddleGAN-develop/psp_output/yingbb2/dst.npy' --output_path 'D://PycharmProjects/PaddleGAN-develop/final_output/yingbb2' --model_type ffhq-config-f --size 1024 --style_dim 512 --n_mlp 8 --channel_multiplier 2 --direction_name gender --direction_offset 5

python -u tools/first-order-demo.py --driving_video 'D://PycharmProjects/PaddleGAN-develop/驱动视频.MOV' --source_image 'D://PycharmProjects/PaddleGAN-develop/final_output/yingbb2/dst.editing.png' --relative --adapt_scale --output 'D://PycharmProjects/PaddleGAN-develop/异性的你动态版' --image_size 512 --face_enhancement



======================== 动漫化 ========================


python applications/tools/animeganv2.py --input_image 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000221754.jpg'


python applications/tools/animeganv2.py --input_image 'D://PycharmProjects/PaddleGAN-develop/yingbb23.png'

python applications/tools/animeganv2.py --input_image epoch00000227_seedidx00000002.png

python tools/main.py --config-file configs/animeganv2.yaml --evaluate-only


======================== lap_style ========================
电路板
python applications/tools/lapstyle.py --style circuit --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000221754.jpg'

python applications/tools/lapstyle.py --style circuit --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000013729.jpg'

python applications/tools/lapstyle.py --style circuit --content_img 'D://PycharmProjects/4Dcloth20211017_final_small/image/00000393@10=person_whole_front.png'


布满星星的
python applications/tools/lapstyle.py --style starrynew --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000221754.jpg'

python applications/tools/lapstyle.py --style starrynew --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000013729.jpg'

python applications/tools/lapstyle.py --style starrynew --content_img 'D://PycharmProjects/4Dcloth20211017_final_small/image/00000393@10=person_whole_front.png'




海浪
python applications/tools/lapstyle.py --style ocean --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000221754.jpg'


星星
python applications/tools/lapstyle.py --style stars --content_img 'D://PycharmProjects/Paddle-PPYOLO-master/images/test/000000221754.jpg'


python applications/tools/lapstyle.py --style stars --content_img 'D://GitHub/ppgan/epoch00000227_seedidx00000002.png'




======================== 超分PP-MSVSR ========================

python tools/main.py --config-file configs/msvsr_l_reds.yaml --evaluate-only --load PP-MSVSR-L_reds_x4.pdparams



python tools/main.py --config-file configs/msvsr_reds.yaml --evaluate-only --load PP-MSVSR_reds_x4.pdparams






