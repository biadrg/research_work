# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi


if [ ! -d "./logs/NLinear_pred_len_96_720" ]; then
    mkdir ./logs/NLinear_pred_len_96_720
fi


model_name=NLinear

num_qubits=10
QML_device=default.qubit

root_path_name=./dataset/

for pred_len in 96 720
do
    for seq_len in 48 96 192 336 504 720
    do
        python -u run_longExp.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path weather.csv \
        --model_id weather_$seq_len'_'$pred_len \
        --model $model_name \
        --data custom \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 21 \
        --des 'Exp' \
        --num_qubits $num_qubits \
        --QML_device $QML_device \
        --itr 1 \
        --batch_size 16  \
        --learning_rate 0.0001 >logs/NLinear_pred_len_96_720/$model_name'_'Weather_$seq_len'_'$pred_len.log
    done
done