ATTRI="test" # "brightness","realism","detail"
LEN=32
echo $ATTRI
CUDA_VISIBLE_DEVICES="0" nohup python train_flux.py \
  --dataset_path "./dataset" \
  --output_path "./models/$ATTRI" \
  --attri "$ATTRI" \
  --attri_len $LEN \
  --max_epochs 1 \
  --steps_per_epoch 2000000 \
  --learning_rate 1e-5 \
  --lora_rank 32 \
  --lora_alpha 32 \
  --precision "bf16" \
  --use_gradient_checkpointing \
  --align_to_opensource_format \
  --center_crop \
  > "log/train_$ATTRI.log" 2>&1 &

echo "Training started, check logs with: log/train_$ATTRI.log"
echo "Process started with PID: $!"