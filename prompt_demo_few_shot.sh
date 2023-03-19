CURRENT_DIR=`pwd`

#    ["dicesimi","jaccard","LongestSamestr","MiniEditDistance"]


python3 prompt_demo_few_shot.py \
  --do_train \
  --do_test \
  --use_fewshot \
  --data_type="bm25"


python3 prompt_demo_few_shot.py \
  --do_train \
  --do_test \
  --use_fewshot \
  --data_type="jaccard"


python3 prompt_demo_few_shot.py \
  --do_train \
  --do_test \
  --use_fewshot \
  --data_type="LongestSamestr"


python3 prompt_demo_few_shot.py \
  --do_train \
  --do_test \
  --use_fewshot \
  --data_type="MiniEditDistance"