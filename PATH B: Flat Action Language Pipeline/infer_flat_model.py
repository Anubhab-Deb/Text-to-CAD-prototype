#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def load_model(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return tokenizer, model

def predict(text, tokenizer, model, max_length=256):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Natural language description")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output", default=None, help="Optional file to save action string")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir)
    action_str = predict(args.text, tokenizer, model)
    print("Generated action string:")
    print(action_str)
    if args.output:
        with open(args.output, "w") as f:
            f.write(action_str)
