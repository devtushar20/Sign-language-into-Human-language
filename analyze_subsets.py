import json
import os

base_dir = r"C:\Users\Atul\project\new pro\wlasl-complete"
words_to_check = ["i", "love", "you", "hello", "bye", "yes", 
                  "no", "please", "sorry", "help", "good", "bad",
                  "thank you", "want", "name", "what", "where"]

# Also map "thank you" to "thank" as per dataset convention
check_map = {w: w for w in words_to_check}
check_map["thank you"] = "thank"

subsets = [100, 300, 1000]

# We need the gloss mapping to check indices in nslt_*.json
class_list_path = os.path.join(base_dir, "wlasl_class_list.txt")
idx_to_gloss = {}
if os.path.exists(class_list_path):
    with open(class_list_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                idx_to_gloss[int(parts[0])] = parts[1]

for s in subsets:
    path = os.path.join(base_dir, f"nslt_{s}.json")
    if not os.path.exists(path):
        continue
        
    with open(path, 'r') as f:
        data = json.load(f)
        
    # In nslt_*.json, action[0] is the index (integer) or the gloss string
    found_words = []
    for w in words_to_check:
        target = check_map[w]
        for vid, info in data.items():
            gloss_info = info['action'][0]
            gloss = idx_to_gloss.get(gloss_info, str(gloss_info))
            if gloss.lower() == target.lower():
                found_words.append(w)
                break
    
    print(f"=== nslt_{s}.json ===")
    for w in words_to_check:
        status = "✅ YES" if w in found_words else "❌ NO"
        print(f"{w}: {status}")
    print("-" * 20)
