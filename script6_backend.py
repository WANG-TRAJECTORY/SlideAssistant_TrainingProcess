import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")

class QwenLocalDeployment:
    def __init__(self, model_path="./lora_finetuned", use_4bit=True, device="cuda"):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")


        self.tokenizer = None
        self.model = None
        self.load_model()

    def load_model(self):
        """加载模型和分词器"""
        logging.info("正在加载分词器...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logging.info("分词器加载完成")
        except Exception as e:
            logging.error(f"分词器加载失败: {e}")
            raise

        logging.info("正在加载模型...")

        try:
            if self.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            self.model.to(self.device)
            logging.info("模型加载完成！")
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise

    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """生成响应"""
        messages = [
            {"role": "system", "content": "你是一个专业的深度学习和机器学习助手。"},
            {"role": "user", "content": prompt}
        ]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

def load_text_from_file(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return file.read()

#切分
def split_text_into_chunks(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

st_model = SentenceTransformer('all-MiniLM-L6-v2')

slide_txt_path = "course_slides_text.txt"
full_text = load_text_from_file(slide_txt_path)

slide_chunks = split_text_into_chunks(full_text)

start_time = time.time()
chunk_embeddings = st_model.encode(slide_chunks)
embedding_time = time.time() - start_time

user_query = "什么是padding?"
query_embedding = st_model.encode([user_query])

similarity_scores = cosine_similarity(query_embedding, chunk_embeddings)

top_n = 3
top_n_indexes = similarity_scores.argsort()[0][-top_n:][::-1]
relevant_chunks = [slide_chunks[i] for i in top_n_indexes]

shortened_chunks = [chunk.strip()[:200] for chunk in relevant_chunks]
context = " ".join(shortened_chunks)[:600]

print("Most relevant shortened chunks:")
for chunk in shortened_chunks:
    print(chunk)

prompt = f"请把95%的注意力放在问题本身上，仅将以下上下文作为辅助参考（不要过度依赖上下文）来回答问题。\n上下文: {context}\n问题: {user_query}"

qwen_model = QwenLocalDeployment(model_path="./lora_finetuned", use_4bit=False)
answer = qwen_model.generate_response(prompt)

execution_time = time.time() - start_time

print(f"\nGenerated Answer: {answer}")
print(f"\nTime taken to compute embeddings: {embedding_time:.4f} seconds")
print(f"Total execution time: {execution_time:.4f} seconds")