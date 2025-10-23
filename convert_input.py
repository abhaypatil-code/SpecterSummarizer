import json

# Read original file with "Judgment" field
input_file = "data/val_judg.jsonl"
output_file = "data/val_judg_formatted.jsonl"

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        data = json.loads(line)
        
        # Create new format with "judgment_text" field and "summarize:" prefix
        new_data = {
            "ID": data["ID"],
            "judgment_text": "summarize: " + data["Judgment"]
        }
        
        f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')

print(f"✅ Conversion complete! Created {output_file}")
print(f"   Now run inference with the formatted file.")
