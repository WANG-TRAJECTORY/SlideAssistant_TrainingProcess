
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import os


class QwenFineTuner:
    def __init__(self, model_path="Qwen/Qwen2-0.5B", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        self.model.to(self.device)

    def load_lora_data(self, response_file="deepseek_responses.txt"):
        """Load DeepseekR1 responses for LoRA distillation."""
        if not os.path.exists(response_file):
            raise FileNotFoundError(f"Response file {response_file} not found.")

        with open(response_file, "r", encoding="utf-8") as f:
            content = f.read().split("=" * 80)

        data = []
        for entry in content:
            if "Question:" in entry and "Response:" in entry:
                question = entry.split("Question:")[1].split("Response:")[0].strip()
                response = entry.split("Response:")[1].strip()
                data.append({"question": question, "response": response})

        return Dataset.from_list(data)

    def prepare_lora_dataset(self, dataset):
        """Prepare dataset for LoRA training."""

        def tokenize_function(example):
            prompt = f"<|im_start|>system\nYou are a deep learning expert.<|im_end|>\n<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n{example['response']}<|im_end>"
            encoding = self.tokenizer(
                prompt,
                truncation=True,
                max_length=1024,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze()
            }

        return dataset.map(tokenize_function)

    def lora_distillation(self, dataset):
        """Perform LoRA distillation."""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)

        training_args = TrainingArguments(
            output_dir="./lora_output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            save_strategy="epoch",
            logging_steps=10
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        print("Starting LoRA distillation...")
        trainer.train()
        self.model.save_pretrained("./lora_finetuned")
        self.tokenizer.save_pretrained("./lora_finetuned")
        print("LoRA distillation completed!")

    def unsupervised_pft(self, slide_text_file="course_slides_text.txt"):
        """Unsupervised PFT fine-tuning using course slides."""
        if not os.path.exists(slide_text_file):
            raise FileNotFoundError(f"Slide text file {slide_text_file} not found.")

        with open(slide_text_file, "r", encoding="utf-8") as f:
            texts = f.read().split("=== 第")

        data = [{"text": text.strip()} for text in texts if text.strip()]
        dataset = Dataset.from_list(data)

        def tokenize_function(example):
            encoding = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=512,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": encoding["input_ids"].squeeze()
            }

        dataset = dataset.map(tokenize_function)

        training_args = TrainingArguments(
            output_dir="./LoRA&pft_output",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            learning_rate=1e-5,
            fp16=True,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        print("Starting unsupervised PFT...")
        trainer.train()
        self.model.save_pretrained("./LoRA&pft_finetuned")
        self.tokenizer.save_pretrained("./LoRA&pft_finetuned")
        print("Unsupervised PFT completed!")


# Usage
if __name__ == "__main__":
    fine_tuner = QwenFineTuner()
    lora_dataset = fine_tuner.load_lora_data()
    lora_dataset = fine_tuner.prepare_lora_dataset(lora_dataset)
    fine_tuner.lora_distillation(lora_dataset)
    fine_tuner.unsupervised_pft()
    print("\nScript4 completed!")