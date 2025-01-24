train_LLM=llama3
train_LLM_name=Meta-Llama-3-8B-Instruct
sft=lora
train_type=dpo
GPU=0

for dataset in msmarco_dpo
do
# *************** #
  #saves_ckpt=PLMs/trained_llm/${train_LLM_name}/${train_type}/${sft}
  yaml_ckpt=../yamls/${train_LLM}/export/${dataset}

  if [ ! -d "$yaml_ckpt" ]; then
    mkdir -p "$yaml_ckpt"
    echo "Directory created and script exited."
    exit 0
  fi

  #if [ ! -d "$saves_ckpt" ]; then
   # mkdir -p "$saves_ckpt"
  #fi
# ********************** #
  CUDA_VISIBLE_DEVICES=${GPU} llamafactory-cli export ${yaml_ckpt}/export_${dataset}_2.0.yaml
done
