#!/bin/bash
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

# sh ./experiments/language_model/rtd_dev.sh deberta-v3-xsmall-continue /mnt/shared/vchoang/works/projects/oda/text2sql/code/DeBERTa/exp/data/wikitext-103 512 wiki spm_deberta-v3-base /mnt/shared/vchoang/works/projects/oda/text2sql/code/DeBERTa/exp/models/deberta-v3-xsmall-continue-wikitext103 /mnt/shared/vchoang/works/projects/oda/text2sql/code/DeBERTa/exp/checkpoints/deberta-v3-xsmall/pytorch_model.generator.bin /mnt/shared/vchoang/works/projects/oda/text2sql/code/DeBERTa/exp/checkpoints/deberta-v3-xsmall/pytorch_model.bin

cache_dir=$2
max_seq_length=$3
data_name_prefix=$4
data_dir=${cache_dir}/spm_${data_name_prefix}_$max_seq_length
spm_reuse=$5
generator_ckpt=$7
discriminator_ckpt=$8

function setup_data(){
	mkdir -p $cache_dir
	if [[ "${spm_reuse}" == "spm_deberta-v3-base" ]]; then
		wget -q https://huggingface.co/microsoft/deberta-v3-base/resolve/main/spm.model -O $cache_dir/spm.model
	fi

  mkdir -p $data_dir
	if [[ ! -e  $data_dir/spm_${data_name_prefix}_${max_seq_length}_train.txt ]]; then
	  python ./prepare_data.py -i ${cache_dir}/${data_name_prefix}.train.tokens -o $data_dir/spm_${data_name_prefix}_${max_seq_length}_train.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i ${cache_dir}/${data_name_prefix}.valid.tokens -o $data_dir/spm_${data_name_prefix}_${max_seq_length}_valid.txt --max_seq_length $max_seq_length
		python ./prepare_data.py -i ${cache_dir}/${data_name_prefix}.test.tokens -o $data_dir/spm_${data_name_prefix}_${max_seq_length}_test.txt --max_seq_length $max_seq_length
	fi
}

setup_data

training_obj=RTD
init=$1
tag=$init
output_dir=$6
case ${init,,} in
	deberta-v3-xsmall-continue)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--num_training_steps 100000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--init_generator ${generator_ckpt} \
	--init_discriminator ${discriminator_ckpt} \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-xsmall)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_xsmall.json \
	--warmup 10000 \
	--learning_rate 3e-4 \
	--train_batch_size 64 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-small)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_small.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-base)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_base.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	deberta-v3-large)
	parameters=" --num_train_epochs 1 \
	--model_config rtd_large.json \
	--warmup 10000 \
	--learning_rate 1e-4 \
	--train_batch_size 256 \
	--decoupled_training True \
	--fp16 True "
		;;
	*)
		echo "usage $0 <Pretrained model configuration>"
		echo "Supported configurations"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 XSmall model with 9M backbone network parameters (12 layers, 256 hidden size) plus 32M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Base model with 81M backbone network parameters (12 layers, 768 hidden size) plus 96M embedding parameters(128k vocabulary size)"
		echo "deberta-v3-xsmall - Pretrained DeBERTa v3 Large model with 288M backbone network parameters (24 layers, 1024 hidden size) plus 128M embedding parameters(128k vocabulary size)"
		exit 0
		;;
esac

mkdir -p $output_dir
python -m DeBERTa.apps.run --model_config $output_dir/config.json  \
	--tag $tag \
	--do_train \
	--num_training_steps 1000000 \
	--max_seq_len $max_seq_length \
	--dump 10000 \
	--task_name ${training_obj} \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir $output_dir \
	$parameters
