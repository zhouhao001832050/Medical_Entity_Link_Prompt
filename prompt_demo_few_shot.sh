CURRENT_DIR=`pwd`


python3 prompt_demo_few_shot.py \
  --do_train \
  --do_test \
  --use_fewshot \
  --data_type="bm25"