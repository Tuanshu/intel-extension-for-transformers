from transformers import AutoTokenizer, TextStreamer
from intel_extension_for_transformers.transformers import AutoModelForCausalLM

#from transformers  import AutoModelForCausalLM

# NOTE: 好像沒看到cached_file in intel_extension_for_transformers.transformers, 所以先用transformers下載
# 要求bitsandbytes 所以下載了

#model_name = "Intel/neural-chat-7b-v3-1"     # Hugging Face model_id or local model
#model_name = "/data/.cache/huggingface/hub/models--Intel--neural-chat-7b-v3-1/snapshots/6dbd30b1d5720fde2beb0122084286d887d24b40"     # Hugging Face model_id or local model

model_name = "Qwen/Qwen-14B-Chat"   
#model_name = "THUDM/chatglm3-6b"   
#model_name = "/data/.cache/huggingface/hub/models--THUDM--chatglm3-6b/snapshots/b098244a71fbe69ce149682d9072a7629f7e908c"   

#prompt = "Once upon a time, there existed a little girl,"
#prompt = "\n______________________\nContext:\n想了解薪資的考勤區間如何計算: 前月16日至當月15日\n\n\n______________________\n\nContext:\n想 了解薪資的餐費扣款區間如何計算: 前前月21到前月20\n\n______________________\nYou are an expert Q&A system that is trusted around the world.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. If a user requests actions beyond providing information, politely inform them that you are unable to do so.\n3. Answer with the same languege as the user's query.\n4. Use Markdown format for all hyperlinks in your responses: [Link Text](URL).\n5. If the user's question involves information not present in the context, respond kindly that you do not know.\n6. Always use the provided context information to answer a question. Never rely on prior knowledge.\n USER: 請告訴我薪資的考勤區間如何計算"
prompt = "請問郭台銘是誰?"
prompt = "\n______________________\nContext:\n各部門可否彈性上班: 針對各部 門能否彈性上班的問題，部分單位沒有執行彈性上班，若有相關疑問請您與單位主管確認。\n\n\n______________________\n\nContext:\n請問居家上班的可行性: 關於居家上班的可能性，目前比照集團政策，未開放居家上班。\n\n______________________\nYou are an expert Q&A system that is trusted around the world.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. If a user requests actions beyond providing information, politely inform them that you are unable to do so.\n3. Answer with the same languege as the user's query.\n4. Use Markdown format for all hyperlinks in your responses: [Link Text](URL).\n5. If the user's question involves information not present in the context, respond kindly that you do not know.\n6. Always use the provided context information to answer a question. Never rely on prior knowledge. 請問彈性上下班規定? \n?"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
inputs = tokenizer(prompt, return_tensors="pt").input_ids
streamer = TextStreamer(tokenizer)

print('[ts] tokenizer ok')

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)#, load_in_4bit=True)#, load_in_4bit=True) #, load_in_8bit=True

print(f'[ts] model ok, type(model)={type(model)}')

outputs = model.generate(inputs, streamer=streamer, max_new_tokens=300)

print(f'[ts] check outputs={outputs}')

# chatglm3:
# error loading model: model.cpp: tensor 'transformer.word_embeddings.weight' is missing from model
# model_init_from_file: failed to load model