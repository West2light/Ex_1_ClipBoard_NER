import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

def test_ner():
    model_path = os.path.abspath("model/ner_xlmr_clipboard")
    print(f"Loading from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    id2label = model.config.id2label
    
    print("Labels:", id2label)
    
    text = "Nguyễn Văn A [PHONE] 45 Lê Lợi Q1 HCM"
    encoding = tokenizer(
        text, 
        return_offsets_mapping=True, 
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"][0]
    offsets = encoding["offset_mapping"][0].tolist()
    
    with torch.no_grad():
        outputs = model(input_ids=encoding["input_ids"])
        
    predictions = torch.argmax(outputs.logits[0], dim=-1).tolist()
    
    for token_id, offset, pred_id in zip(input_ids, offsets, predictions):
        token_str = tokenizer.convert_ids_to_tokens(token_id.item())
        label = id2label.get(pred_id, "O")
        print(f"{token_str:15} {offset} {label}")

if __name__ == "__main__":
    test_ner()
