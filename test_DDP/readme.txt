

CUDA_VISIBLE_DEVICES=0
python test2_01_FullyConnectedLayer_grad_pytorch.py

python test2_01_FullyConnectedLayer_grad_diff_pytorch.py



CUDA_VISIBLE_DEVICES=0,1
python test2_01_FullyConnectedLayer_grad_pytorch.py

python test2_01_FullyConnectedLayer_grad_diff_pytorch.py











