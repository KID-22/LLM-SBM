train_LLM=llama3
train_LLM_name=Meta-Llama-3-8B-Instruct
sft=lora

GPU=0

for alpha in 2.0
do
  for dataset in msmarco_dpo
  do
  # **************** #
    saves_ckpt=../saves/${train_LLM_name}/${sft}/dpo/${dataset}
    # **************** #
    yaml_ckpt=../yamls/${train_LLM}/${sft}/${dataset}
    # **************** #
    loggers_ckpt=../loggers/${train_LLM}/${sft}/${dataset}

    if [ ! -d "$yaml_ckpt" ]; then
      mkdir -p "$yaml_ckpt"
      echo "Directory created and script exited."
      exit 0
    fi

    if [ ! -d "$saves_ckpt" ]; then
      mkdir -p "$saves_ckpt"
    fi

    if [ ! -d "$loggers_ckpt" ]; then
      mkdir -p "$loggers_ckpt"
    fi
  # **************** #
    
    CUDA_VISIBLE_DEVICES=${GPU} llamafactory-cli train ${yaml_ckpt}/msmarco_dpo.yaml > ${loggers_ckpt}/dpo_${dataset}_${alpha}.log
  done
done

