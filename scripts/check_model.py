from transformers import AutoTokenizer, AutoConfig

model_name = "smilegate-ai/kor_unsmile"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

print("tokenizer.model_max_length:", tokenizer.model_max_length)
print("config.max_position_embeddings:", getattr(config, "max_position_embeddings", None))