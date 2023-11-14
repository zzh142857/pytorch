#!/bin/bash

declare -A model_batch_sizes
model_batch_sizes["stable_diffusion_unet"]=8
model_batch_sizes["hf_T5"]=22
model_batch_sizes["hf_T5_base"]=8
model_batch_sizes["hf_T5_large"]=8
model_batch_sizes["hf_Whisper"]=96
model_batch_sizes["llama_v2_7b_16h"]=8
model_batch_sizes["nanogpt"]=896

for model in "${!model_batch_sizes[@]}"; do
    batch_size="${model_batch_sizes[$model]}"
		# echo "Running accuracy for $model"
		# https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 PYTHONPATH=./../benchmark HUGGING_FACE_HUB_TOKEN=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 python benchmarks/dynamo/torchbench.py --accuracy --no-translation-validation --training --amp --backend inductor --device cuda --output inductor_with_cudagraphs_torchbench_amp_training_cuda_accuracy.csv --only $model --ddp --multiprocess --batch_size=$batch_size # 2>&1 | pastry -t "A100 80GB $model ddp accuracy"
    echo "Running perf benchmark for $model"
    cmd="https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 PYTHONPATH=./../benchmark HUGGING_FACE_HUB_TOKEN=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 python benchmarks/dynamo/torchbench.py --performance --cold-start-latency --training --amp --backend inductor --device cuda --output inductor_with_cudagraphs_torchbench_amp_training_cuda_performance.csv --only $model --ddp --multiprocess --batch_size=$batch_size"

    if [ "$model" == "stable_diffusion_unet" ] || [ "$model" == "llama_v2_7b_16h" ]; then
      cmd+=" --no-optimize-ddp"
    fi

    cmd+=" --export-profiler-trace --repeat 5"

    #{
    #  echo $cmd
    #  eval $cmd
    #} 2>&1 | pastry -t "$model DDP perf"
    echo $cmd
    eval $cmd
done
