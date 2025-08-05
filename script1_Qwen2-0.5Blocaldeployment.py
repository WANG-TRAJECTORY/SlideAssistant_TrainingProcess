import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import warnings
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")


class QwenLocalDeployment:
    def __init__(self, model_path="Qwen/Qwen2-0.5B", use_4bit=True):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")

        # 检查 GPU 内存
        if torch.cuda.is_available():
            logging.info(f"GPU: {torch.cuda.get_device_name()}")
            logging.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB")

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
            logging.info("模型加载完成！")
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise

    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """生成响应"""
        formatted_prompt = f"<|im_start|>system\n你是一个专业的深度学习和机器学习助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

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

    def test_neural_network_questions(self):
        """测试神经网络相关问题"""
        test_questions = [
            "什么是反向传播算法？请详细解释其工作原理。",
            "解释梯度下降算法的基本思想和实现步骤。",
            "卷积神经网络(CNN)的核心组件有哪些？",
            "什么是过拟合？如何防止过拟合？",
            "解释RNN和LSTM的区别。"
        ]

        logging.info("\n=== 神经网络课程问题测试 ===")
        for i, question in enumerate(test_questions, 1):
            logging.info(f"\n问题 {i}: {question}")
            logging.info("-" * 50)
            response = self.generate_response(question)
            logging.info(f"回答: {response}")
            logging.info("=" * 80)

    def save_model(self, save_path="./qwen2-0.5b"):
        """保存模型和分词器到指定路径"""
        try:
            os.makedirs(save_path, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logging.info(f"模型和分词器已保存到 {save_path}")
            print(f"模型已保存到 {save_path}")
        except Exception as e:
            logging.error(f"模型保存失败: {e}")
            raise


if __name__ == "__main__":
    try:
        # 初始化Qwen模型
        logging.info("初始化Qwen2-0.5B模型...")
        qwen_model = QwenLocalDeployment(use_4bit=False)

        # 测试基础问题
        qwen_model.test_neural_network_questions()

        # 保存模型到根目录
        qwen_model.save_model()

        logging.info("\n第一段脚本执行完成！")
    except Exception as e:
        logging.error(f"脚本执行失败: {e}")