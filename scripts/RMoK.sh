for window in 5 10 20 48 96; do
    python run.py \
        --model_id "HS" \
        --is_training 1 \
        --model "RMoK" \
        --root_path "./dataset/" \
        --data_path "HS.csv" \
        --log_dir "./logs/" \
        --log_name "RMoK.txt" \
        --data "HS" \
        --features "M" \
        --seq_len $window \
        --pred_len $window \
        --enc_in 13 \
        --dec_in 13 \
        --d_model 32 \
        --itr 1 \
        --train_epochs 100 \
        --batch_size 256 \
        --patience 3 \
        --learning_rate 0.001 \
        --des "Exp" \
        --lradj "type3" \
        --use_multi_scale "false" \
        --small_kernel_merged "false" \
        --gpu 2
done