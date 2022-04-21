

CUDA_VISIBLE_DEVICES=0
python test2_01_grad_pytorch.py

python test2_01_grad_diff_pytorch.py



CUDA_VISIBLE_DEVICES=0,1
python test2_01_grad_pytorch.py

python test2_01_grad_diff_pytorch.py


------------------------ Paddle 单卡 ------------------------
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu


CUDA_VISIBLE_DEVICES=0
python test2_01_grad_paddle.py


python test2_01_grad_diff_paddle.py



------------------------ Paddle的DP ------------------------
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu


CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch test2_01_grad_paddle_dp.py


python test2_01_grad_diff_paddle.py










