CUDA_VISIBLE_DEVICES=2 python -m main_probe --accum_iter 1 \
    --batch_size 64 \
    --epochs 50 \
    --weight_decay 0.05 \
    --lr 2e-4 \
    --warmup_epochs 0 \
    --output_dir log/output_probe_TAG \
    --log_dir log/output_probe_TAG \
    --pooling cls \
    --dataset rough \
    --load_model \
    --num_workers 8 \
    --load_path /data/develop/smallz/TLV-Core/logs/TAG/checkpoints/epoch_latest.pt \
    # --load_path /data/develop/smallz/TLV-Core/logs/SSVTP/checkpoints/epoch_latest.pt \
    
    