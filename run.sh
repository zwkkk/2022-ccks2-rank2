# 1. 数据集制作
time=$(date "+%Y%m%d-%H%M%S")
echo "${time}"
echo "doing 1"
cd data
python build_text_input.py
cd ..

# 2. R2D2_cross pretrain+train
time=$(date "+%Y%m%d-%H%M%S")
echo "${time}"
echo "doing 2"
cd R2D2_cross
sh run.sh
cd ..

# 3. R2D2_distill train+predict
time=$(date "+%Y%m%d-%H%M%S")
echo "${time}"
echo "doing 3"
cd R2D2_distill
sh run.sh
cd ..

# submit
cp ./R2D2_distill/outputs/zwk_result.jsonl ./submit/result
cd submit/result
zip result.zip similarity.py zwk_result.jsonl
cd ..
cp -r result/result.zip ./
rm -rf result
