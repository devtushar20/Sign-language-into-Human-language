"""
create_custom_subset.py
Extracts the 30 target signs from WLASL_v0.3.json into custom_subset.json
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WLASL_JSON_PATH, CUSTOM_SUBSET_JSON, CUSTOM_SIGNS

full_path = WLASL_JSON_PATH
target_path = CUSTOM_SUBSET_JSON
signs = [s.lower() for s in CUSTOM_SIGNS]

if not os.path.exists(full_path):
    print(f"Error: Full dataset not found at {full_path}")
    print("Looking for WLASL_v0.3.json in the dataset folder...")
    exit(1)

with open(full_path, 'r') as f:
    data = json.load(f)

custom_data = {}
found_signs = []

for entry in data:
    gloss = entry['gloss'].lower()
    if gloss in signs:
        if gloss not in found_signs:
            found_signs.append(gloss)
        for inst in entry['instances']:
            video_id = inst['video_id']
            custom_data[video_id] = {
                'subset': inst['split'],
                'action': [gloss, inst.get('frame_start', 0), inst.get('frame_end', -1)]
            }

with open(target_path, 'w') as f:
    json.dump(custom_data, f, indent=4)

not_found = [s for s in signs if s not in found_signs]
print(f"Saved {len(custom_data)} video entries to {target_path}")
print(f"Signs found ({len(found_signs)}): {sorted(found_signs)}")
if not_found:
    print(f"Signs NOT found in WLASL: {not_found}")
