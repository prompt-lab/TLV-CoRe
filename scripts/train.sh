# CACHE_DIR="/home/chenning/opensource_models/"
ANNOTATION="dataset/train/touch100k/data_list.json"
# ANNOTATION="dataset/train/SSVTP/data_list.json"
# # add NCCL_P2P_DISABLE=1
# TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node 2 --master_port 29807 \
#     -m main  \
#     --do_train --beta-init 0.9 --span 0.9 --inte-type "above" --decay-type "linear"\
#     --train-data ${ANNOTATION} \
#     --clip_type "tlv" \
#     --init-temp 0.07 --learn-temp \
#     --model "ViT-L-14" \
#     --lock_text \
#     --use_shared_adapter --adapter_shared_dim 32 \
#     --lr 2e-4 --coef-lr 1e-3 \
#     --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
#     --num-frames 1 --force-patch-dropout 0.5 \
#     --epochs 12 --batch-size 96 --accum-freq 1 --warmup 200 \
#     --precision "amp" --workers 4 \
#     --save-frequency 1 --log-every-n-steps 100 --report-to "tensorboard" --resume "latest" \
#     --do_eval \
#     --val_t_cls_data "Touch_and_Go" \
#     --cls_mode "material" \
#     --save-most-recent



# 单卡运行
# TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 python -m main \
    --num-sensors 5 \
    --do_train --beta-init 0.9 --span 0.9 --inte-type "above" --decay-type "linear" \
    --train-data ${ANNOTATION} \
    --clip_type "tlv" \
    --init-temp 0.07 --learn-temp \
    --model "ViT-L-14" \
    --lock_text \
    --use_shared_adapter --adapter_shared_dim 32 \
    --lr 2e-4 --coef-lr 1e-3 \
    --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
    --num-frames 1 --force-patch-dropout 0.5 \
    --epochs 12 --batch-size 128 --accum-freq 1 --warmup 200 \
    --precision "amp" --workers 2 \
    --save-frequency 1 --log-every-n-steps 100 --report-to "tensorboard" --resume "latest" \
    --do_eval \
    --val_t_cls_data "Touch_and_Go" \
    --cls_mode "material" \
    --save-most-recent


# CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node 2 --master_port 29807 \
#     -m main  \
#     --do_train --beta-init 0.9 --span 0.9 --inte-type "above" --decay-type "linear"\
#     --train-data ${ANNOTATION} \
#     --clip_type "tlv" \
#     --init-temp 0.07 --learn-temp \
#     --model "ViT-L-14" \
#     --lock_text \
#     --convert_to_lora --lora_r 16 --lora_alpha 16 \
#     --lr 2e-4 --coef-lr 1e-3 \
#     --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
#     --num-frames 1 --force-patch-dropout 0.5 \
#     --epochs 12 --batch-size 96 --accum-freq 1 --warmup 200 \
#     --precision "amp" --workers 4 \
#     --save-frequency 1 --log-every-n-steps 100 --report-to "tensorboard" --resume "latest" \
#     --do_eval \
#     --val_t_cls_data "Touch_and_Go" \
#     --cls_mode "material" \
#     --save-most-recent




# ANNOTATION="dataset/train/SSVTP/data_list.json"
# TORCH_DISTRIBUTED_DEBUG=DETAIL HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# CUDA_VISIBLE_DEVICES=2 python -m main \
#     --do_train --beta-init 0.9 --span 0.9 --inte-type "above" --decay-type "linear" \
#     --train-data ${ANNOTATION} \
#     --clip_type "tlv" \
#     --init-temp 0.07 --learn-temp \
#     --model "ViT-L-14" \
#     --lock_text \
#     --convert_to_lora --lora_r 16 --lora_alpha 16 \
#     --lr 2e-4 --coef-lr 1e-3 \
#     --beta1 0.9 --beta2 0.98 --wd 0.2 --eps 1e-6 \
#     --num-frames 1 --force-patch-dropout 0.5 \
#     --epochs 12 --batch-size 32 --accum-freq 1 --warmup 200 \
#     --precision "amp" --workers 16 \
#     --save-frequency 1 --log-every-n-steps 100 --report-to "tensorboard" --resume "latest" \
#     --do_eval \
#     --val_t_cls_data "Touch_and_Go" \
#     --cls_mode "material" \
#     --save-most-recent

