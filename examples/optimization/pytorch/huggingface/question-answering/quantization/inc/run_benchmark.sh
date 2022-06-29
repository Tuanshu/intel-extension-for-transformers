#!/bin/bash
set -x

function main {

  init_params "$@"
  run_benchmark

}

# init params
function init_params {
  iters=100
  batch_size=16
  tuned_checkpoint=saved_results
  for var in "$@"
  do
    case $var in
      --topology=*)
          topology=$(echo $var |cut -f2 -d=)
      ;;
      --dataset_location=*)
          dataset_location=$(echo $var |cut -f2 -d=)
      ;;
      --input_model=*)
          input_model=$(echo $var |cut -f2 -d=)
      ;;
      --mode=*)
          mode=$(echo $var |cut -f2 -d=)
      ;;
      --batch_size=*)
          batch_size=$(echo $var |cut -f2 -d=)
      ;;
      --iters=*)
          iters=$(echo ${var} |cut -f2 -d=)
      ;;
      --int8=*)
          int8=$(echo ${var} |cut -f2 -d=)
      ;;
      --config=*)
          tuned_checkpoint=$(echo $var |cut -f2 -d=)
      ;;
      *)
          echo "Error: No such parameter: ${var}"
          exit 1
      ;;
    esac
  done

}


# run_benchmark
function run_benchmark {
    extra_cmd=''
    MAX_SEQ_LENGTH=384

    if [[ ${mode} == "accuracy" ]]; then
        mode_cmd=" --accuracy_only"
    elif [[ ${mode} == "benchmark" ]]; then
        mode_cmd=" --benchmark "
    else
        echo "Error: No such mode: ${mode}"
        exit 1
    fi

    if [ "${topology}" = "distilbert_base_squad_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
    elif [ "${topology}" = "distilbert_base_squad_dynamic" ]; then
        DATASET_NAME="squad"
        model_name_or_path="distilbert-base-uncased-distilled-squad"
    elif [ "${topology}" = "bert_large_SQuAD_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="bert-large-uncased-whole-word-masking-finetuned-squad"
    elif [ "${topology}" = "roberta_base_SQuAD2_static" ]; then
        DATASET_NAME="squad"
        model_name_or_path="deepset/roberta-base-squad2"
    fi

    if [[ ${int8} == "true" ]]; then
        extra_cmd=$extra_cmd" --int8"
        model_name_or_path=${tuned_checkpoint}
    fi
    echo $extra_cmd

    python -u run_qa.py \
        --model_name_or_path ${model_name_or_path} \
        --dataset_name ${DATASET_NAME} \
        --do_eval \
        --max_seq_length ${MAX_SEQ_LENGTH} \
        --per_device_eval_batch_size ${batch_size} \
        --output_dir ./tmp/benchmark_output \
        --overwrite_output_dir \
        --overwrite_cache \
        --no_cuda \
        ${mode_cmd} \
        ${extra_cmd}
}

main "$@"
