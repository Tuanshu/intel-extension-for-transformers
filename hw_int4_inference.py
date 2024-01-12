from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

#from transformers  import AutoModelForCausalLM

# NOTE: 好像沒看到cached_file in intel_extension_for_transformers.transformers, 所以先用transformers下載
# 要求bitsandbytes 所以下載了

#model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
model_name = "/data/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-1/snapshots/6dbd30b1d5720fde2beb0122084286d887d24b40"     # Hugging Face model_id or local model


prompt = "Once upon a time, there existed a little girl,"
#prompt = "\n______________________\nContext:\n想了解薪資的考勤區間如何計算: 前月16日至當月15日\n\n\n______________________\n\nContext:\n想 了解薪資的餐費扣款區間如何計算: 前前月21到前月20\n\n______________________\nYou are an expert Q&A system that is trusted around the world.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. If a user requests actions beyond providing information, politely inform them that you are unable to do so.\n3. Answer with the same languege as the user's query.\n4. Use Markdown format for all hyperlinks in your responses: [Link Text](URL).\n5. If the user's question involves information not present in the context, respond kindly that you do not know.\n6. Always use the provided context information to answer a question. Never rely on prior knowledge.\n USER: 請告訴我薪資的考勤區間如何計算"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

print('[ts] tokenizer ok')

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True) #, load_in_4bit=True

print('[ts] model ok')

outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)