gpu=$1

# CUDA_VISIBLE_DEVICES=$gpu python hf_train_bart.py --do_train --do_segment --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-segment-genkpsum

CUDA_VISIBLE_DEVICES=$gpu python hf_train_bart.py --do_train --gen_keyphrase_summary --output_dir=save/bart-large-xsum-samsum-genkpsum  --test_target_max_len=400 --target_max_len=400 --train_batch_size 2 --eval_batch_size 2 --validation_timing 5000
