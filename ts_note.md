故意把intel_extension_for_transformers -> intel_extension_for_transformers_local

這樣才會用pip裡面的

pip install時有build

這沒有 模型的graph需要build


python  examples/huggingface/pytorch/text-generation/inference/run_generation.py -m Qwen/Qwen-14B-Chat


python  examples/huggingface/pytorch/text-generation/inference/run_generation.py -m THUDM/chatglm3-6b

