exp="aotplus"
# exp="debug"
gpu_num="4"
devices="4,5,6,7"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
# model="swinb_aotl"
	
stage="pre_vost"
CUDA_VISIBLE_DEVICES=${devices} python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num}

dataset="vost"
split="val"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 1.1 1.2 0.9 0.8