#!/usr/bin/env python3
"""
ZIM CODER MAESTRO - The Ultimate Free Open-Source Coding Model
Trained to solve ANY coding problem anywhere in the world!
"""

print("🚀 Installing dependencies...")
!pip install -q transformers datasets accelerate bitsandbytes peft huggingface_hub torch torchvision torchaudio --upgrade
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q safetensors

import os
import torch
import gc
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd

# Clear memory
torch.cuda.empty_cache()
gc.collect()

print("✅ Dependencies installed!")

# ==================== CONFIGURATION ====================
class ZimCoderConfig:
    # Using DeepSeek Coder 1.3B - Best free model that fits in Colab
    BASE_MODEL = "deepseek-ai/deepseek-coder-1.3b-base"
    
    # Training Parameters
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 3
    MAX_LENGTH = 2048
    
    # Quantization
    USE_4BIT = True
    
    # Output
    OUTPUT_DIR = "zim-coder-maestro"
    SAVE_STEPS = 100
    LOGGING_STEPS = 25

# ==================== DOWNLOAD REAL DATASETS ====================

def load_real_training_data():
    """Download and prepare REAL coding datasets"""
    print("📦 Downloading REAL coding datasets...")
    
    datasets = []
    
    # 1. CodeAlpaca 20k - PROVEN DATASET
    print("📥 Downloading CodeAlpaca-20k...")
    try:
        codealpaca = load_dataset("sahil2801/CodeAlpaca-20k", split="train[:15000]")
        
        def format_codealpaca(example):
            instruction = example['instruction']
            input_text = example['input']
            output = example['output']
            
            if input_text.strip():
                return f"<|im_start|>user\n{instruction}\nInput: {input_text}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
            else:
                return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
        
        codealpaca = codealpaca.map(lambda x: {"text": format_codealpaca(x)})
        datasets.append(codealpaca)
        print(f"✅ CodeAlpaca: {len(codealpaca)} samples")
    except Exception as e:
        print(f"⚠️ CodeAlpaca failed: {e}")

    # 2. Python Code Instructions - REAL PYTHON CODE
    print("📥 Downloading Python Code Instructions...")
    try:
        python_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:10000]")
        
        def format_python(example):
            return f"<|im_start|>user\nWrite Python code: {example['instruction']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
        
        python_code = python_code.map(lambda x: {"text": format_python(x)})
        datasets.append(python_code)
        print(f"✅ Python Code: {len(python_code)} samples")
    except Exception as e:
        print(f"⚠️ Python Code failed: {e}")

    # 3. Multi-language Code Examples
    print("📥 Downloading Multi-language Code...")
    try:
        # Create multi-language examples
        multi_lang_examples = [
            # Python
            "<|im_start|>user\nWrite a Python function to calculate factorial<|im_end|>\n<|im_start|>assistant\ndef factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)<|im_end|>",
            
            # JavaScript
            "<|im_start|>user\nCreate a React component for a login form<|im_end|>\n<|im_start|>assistant\nfunction LoginForm() {\n  return (\n    <div>\n      <input type=\"text\" placeholder=\"Username\" />\n      <input type=\"password\" placeholder=\"Password\" />\n      <button>Login</button>\n    </div>\n  );\n}<|im_end|>",
            
            # Java
            "<|im_start|>user\nWrite Java code for binary search<|im_end|>\n<|im_start|>assistant\npublic class BinarySearch {\n    public static int binarySearch(int[] arr, int target) {\n        int left = 0, right = arr.length - 1;\n        while (left <= right) {\n            int mid = left + (right - left) / 2;\n            if (arr[mid] == target) return mid;\n            if (arr[mid] < target) left = mid + 1;\n            else right = mid - 1;\n        }\n        return -1;\n    }\n}<|im_end|>",
            
            # HTML/CSS
            "<|im_start|>user\nCreate a responsive navbar with HTML and CSS<|im_end|>\n<|im_start|>assistant\n<nav class=\"navbar\">\n  <div class=\"nav-brand\">MySite</div>\n  <ul class=\"nav-links\">\n    <li><a href=\"#home\">Home</a></li>\n    <li><a href=\"#about\">About</a></li>\n  </ul>\n</nav>\n\n<style>\n.navbar { display: flex; justify-content: space-between; padding: 1rem; }\n@media (max-width: 768px) { .navbar { flex-direction: column; } }\n</style><|im_end|>",
            
            # SQL
            "<|im_start|>user\nWrite SQL to find customers with most orders<|im_end|>\n<|im_start|>assistant\nSELECT customer_id, COUNT(*) as order_count \nFROM orders \nGROUP BY customer_id \nORDER BY order_count DESC \nLIMIT 10;<|im_end|>"
        ]
        
        # Add more examples programmatically
        languages = {
            'Python': [
                "Write a function to reverse a string",
                "Create a class for a Bank Account",
                "Write a script to read CSV file",
                "Create a Flask web server",
                "Write code to connect to MySQL database"
            ],
            'JavaScript': [
                "Create a function to validate email",
                "Write async code to fetch API data",
                "Create a shopping cart class",
                "Write code for form validation",
                "Create a timer countdown"
            ],
            'Java': [
                "Write a Spring Boot controller",
                "Create a HashMap implementation",
                "Write multithreading code",
                "Create a REST API endpoint",
                "Write file I/O operations"
            ]
        }
        
        for lang, prompts in languages.items():
            for prompt in prompts:
                multi_lang_examples.append(
                    f"<|im_start|>user\nWrite {lang} code: {prompt}<|im_end|>\n<|im_start|>assistant\n# {lang} code for: {prompt}\n# Implementation goes here...<|im_end|>"
                )
        
        multi_lang_dataset = Dataset.from_dict({"text": multi_lang_examples})
        datasets.append(multi_lang_dataset)
        print(f"✅ Multi-language: {len(multi_lang_examples)} samples")
    except Exception as e:
        print(f"⚠️ Multi-language failed: {e}")

    # 4. Zimbabwe-specific coding examples
    print("🇿🇼 Adding Zimbabwe-specific code examples...")
    zim_examples = [
        # Zimbabwe financial applications
        "<|im_start|>user\nWrite Python code for ZWL to USD currency converter for Zimbabwe<|im_end|>\n<|im_start|>assistant\ndef zwl_to_usd(zwl_amount, exchange_rate):\n    \"\"\"Convert Zimbabwe Dollars to USD\"\"\"\n    return zwl_amount / exchange_rate\n\ndef usd_to_zwl(usd_amount, exchange_rate):\n    \"\"\"Convert USD to Zimbabwe Dollars\"\"\"\n    return usd_amount * exchange_rate<|im_end|>",
        
        # Agriculture applications
        "<|im_start|>user\nCreate a Python class for Zimbabwe farm management<|im_end|>\n<|im_start|>assistant\nclass ZimbabweFarm:\n    def __init__(self, name, location, crop_type, hectares):\n        self.name = name\n        self.location = location  # e.g., 'Mashonaland', 'Matabeleland'\n        self.crop_type = crop_type  # e.g., 'maize', 'tobacco', 'cotton'\n        self.hectares = hectares\n    \n    def calculate_yield(self, yield_per_hectare):\n        return self.hectares * yield_per_hectare\n    \n    def get_province(self):\n        zimbabwe_provinces = ['Harare', 'Bulawayo', 'Manicaland', 'Mashonaland']\n        for province in zimbabwe_provinces:\n            if province.lower() in self.location.lower():\n                return province\n        return 'Unknown'<|im_end|>",
        
        # Mobile money (EcoCash)
        "<|im_start|>user\nWrite JavaScript for EcoCash mobile money integration<|im_end|>\n<|im_start|>assistant\nclass EcoCashPayment {\n    constructor(apiKey) {\n        this.apiKey = apiKey;\n        this.baseURL = 'https://api.ecocash.co.zw';\n    }\n    \n    async sendMoney(phoneNumber, amount, reference) {\n        const response = await fetch(`${this.baseURL}/send`, {\n            method: 'POST',\n            headers: {\n                'Authorization': `Bearer ${this.apiKey}`,\n                'Content-Type': 'application/json'\n            },\n            body: JSON.stringify({\n                phone: phoneNumber,\n                amount: amount,\n                reference: reference\n            })\n        });\n        return await response.json();\n    }\n    \n    async checkBalance(phoneNumber) {\n        const response = await fetch(`${this.baseURL}/balance/${phoneNumber}`, {\n            headers: {'Authorization': `Bearer ${this.apiKey}`}\n        });\n        return await response.json();\n    }\n}<|im_end|>",
        
        # African e-commerce
        "<|im_start|>user\nCreate a Django model for African online marketplace<|im_end|>\n<|im_start|>assistant\nfrom django.db import models\n\nclass AfricanMarketplaceProduct(models.Model):\n    PRODUCT_CATEGORIES = [\n        ('agriculture', 'Agriculture'),\n        ('handicrafts', 'Handicrafts'),\n        ('textiles', 'Textiles'),\n        ('art', 'African Art'),\n    ]\n    \n    name = models.CharField(max_length=200)\n    description = models.TextField()\n    price = models.DecimalField(max_digits=10, decimal_places=2)\n    category = models.CharField(max_length=50, choices=PRODUCT_CATEGORIES)\n    seller_country = models.CharField(max_length=50)  # Zimbabwe, South Africa, etc.\n    available = models.BooleanField(default=True)\n    \n    def __str__(self):\n        return f\"{self.name} from {self.seller_country}\"<|im_end|>"
    ]
    
    zim_dataset = Dataset.from_dict({"text": zim_examples})
    datasets.append(zim_dataset)
    print(f"✅ Zimbabwe Examples: {len(zim_examples)} samples")

    # Combine all datasets
    if datasets:
        print("🔄 Combining all datasets...")
        combined_dataset = concatenate_datasets(datasets)
        
        print(f"🎉 TOTAL TRAINING DATA: {len(combined_dataset)} examples")
        return combined_dataset
    else:
        raise Exception("❌ No datasets were loaded successfully!")

# ==================== DOWNLOAD AND SETUP MODEL ====================

def setup_model():
    """Download and setup DeepSeek Coder model"""
    print("🛠️ Downloading DeepSeek Coder model...")
    
    # Configure 4-bit quantization to fit in Colab
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Download tokenizer
    print("📥 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ZimCoderConfig.BASE_MODEL)
    
    # Add special tokens for chat format
    tokenizer.add_special_tokens({
        'pad_token': '<|pad|>',
        'eos_token': '<|im_end|>',
        'bos_token': '<|im_start|>'
    })
    
    # Download model with quantization
    print("📥 Downloading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        ZimCoderConfig.BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✅ Model downloaded: {ZimCoderConfig.BASE_MODEL}")
    print(f"✅ Model device: {model.device}")
    print(f"✅ Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer

# ==================== TRAINING SETUP ====================

def setup_training(model, tokenizer, dataset):
    """Setup the trainer with training arguments"""
    print("⚙️ Setting up training...")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=ZimCoderConfig.MAX_LENGTH,
        )
    
    print("🔧 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=ZimCoderConfig.OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=ZimCoderConfig.EPOCHS,
        per_device_train_batch_size=ZimCoderConfig.BATCH_SIZE,
        gradient_accumulation_steps=ZimCoderConfig.GRAD_ACCUM_STEPS,
        learning_rate=ZimCoderConfig.LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=ZimCoderConfig.LOGGING_STEPS,
        save_steps=ZimCoderConfig.SAVE_STEPS,
        save_total_limit=2,
        fp16=True,
        report_to=None,
        dataloader_pin_memory=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )
    
    return trainer

# ==================== TEST THE MODEL ====================

def test_model(model, tokenizer):
    """Test the trained model on various coding problems"""
    print("\n🧪 TESTING ZIM CODER MAESTRO...")
    
    test_cases = [
        # Zimbabwe-specific tests
        "Write Python code for EcoCash payment processing in Zimbabwe",
        "Create a function to calculate maize yield for Zimbabwe farms",
        "Build a Harare restaurant booking system in JavaScript",
        
        # General coding tests
        "Write a Python function to find prime numbers",
        "Create a React component for a weather app",
        "Write SQL queries for an e-commerce database",
        "Implement binary search algorithm in Java",
        "Create a Dockerfile for a Python web application",
        
        # Complex problems
        "Build a machine learning model to predict stock prices",
        "Create a REST API with authentication in Node.js",
        "Write a web scraper for news articles in Python"
    ]
    
    for i, prompt in enumerate(test_cases[:5]):  # Test first 5
        print(f"\n🎯 Test {i+1}: {prompt}")
        print("=" * 60)
        
        # Format prompt
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + 300,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract only the assistant's response
        if "<|im_start|>assistant" in generated:
            response = generated.split("<|im_start|>assistant")[1].split("<|im_end|>")[0].strip()
            print(f"💻 Generated Code:\n{response}")
        else:
            print(f"💻 Response:\n{generated}")
        
        print("-" * 50)

# ==================== MAIN TRAINING FUNCTION ====================

def train_zim_coder_maestro():
    """Main training function"""
    print("🎯 STARTING ZIM CODER MAESTRO TRAINING!")
    print("🇿🇼 Creating the Ultimate Coding Model for Zimbabwe and Beyond!")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name}")
        print(f"✅ GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("❌ No GPU detected! Training will be very slow.")
        return
    
    try:
        # Step 1: Load real datasets
        print("\n📊 STEP 1: Loading training data...")
        dataset = load_real_training_data()
        
        # Step 2: Download and setup model
        print("\n🤖 STEP 2: Downloading DeepSeek Coder model...")
        model, tokenizer = setup_model()
        
        # Step 3: Setup training
        print("\n⚙️ STEP 3: Configuring training...")
        trainer = setup_training(model, tokenizer, dataset)
        
        # Step 4: Start training
        print("\n🚀 STEP 4: Starting training...")
        print("💡 This will take 2-4 hours. Please wait...")
        print("📈 Monitoring progress...")
        
        trainer.train()
        
        # Step 5: Save the model
        print("\n💾 STEP 5: Saving the trained model...")
        trainer.save_model()
        tokenizer.save_pretrained(ZimCoderConfig.OUTPUT_DIR)
        
        # Step 6: Test the model
        print("\n🎯 STEP 6: Testing the model...")
        test_model(model, tokenizer)
        
        # Step 7: Prepare for download
        print("\n📦 STEP 7: Creating download package...")
        !zip -r {ZimCoderConfig.OUTPUT_DIR}.zip {ZimCoderConfig.OUTPUT_DIR}/
        
        print("\n🎉 🎉 🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉 🎉 🎉")
        print("=" * 70)
        print("🇿🇼 ZIM CODER MAESTRO IS READY! 🇿🇼")
        print(f"📁 Model saved to: {ZimCoderConfig.OUTPUT_DIR}")
        print(f"📦 Download file: {ZimCoderConfig.OUTPUT_DIR}.zip")
        
        print("\n🚀 NEXT STEPS:")
        print("1. Download the zip file from Colab")
        print("2. Extract it and use with your API server")
        print("3. Deploy as the BEST FREE CODING MODEL!")
        print("4. Solve coding problems ANYWHERE in the world!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

# ==================== DOWNLOAD FUNCTION ====================

def download_trained_model():
    """Download the trained model"""
    from google.colab import files
    
    zip_file = f"{ZimCoderConfig.OUTPUT_DIR}.zip"
    if os.path.exists(zip_file):
        print(f"📥 Downloading {zip_file}...")
        files.download(zip_file)
    else:
        print("❌ Model not found. Run training first.")

# ==================== RUN THE TRAINING ====================

if __name__ == "__main__":
    # Start the training
    train_zim_coder_maestro()
    
    # Uncomment below to download immediately after training
    # download_trained_model()
