accelerate launch \
    ./image-classification-accelerate-gpu.py \
--seed 0 \
--gradient_accumulation_steps 1 \
--max_epochs 1 \
--network 'resnet152' \
--dataset 'oxfordiiitpet' \
--num_images 1_000 \
--num_classes 1000 \
--batch_size 32 \
--image_size 224 \
--workers 8 \
--persistent_workers \
--learning_rate 2e-4 \
--learning_rate_downscale 3 \
--optim_name 'adamw'
