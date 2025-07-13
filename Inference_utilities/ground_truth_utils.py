import re

def load_ground_truth(label_file):
        gts = []
        with open(label_file, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
    
        for line in lines:
            line = line.strip()
            
            x_match = re.findall(r"x:\s*\[\[([^\]]+)\]\]", line)
            y_match = re.findall(r"y:\s*\[\[([^\]]+)\]\]", line)
    
            if not x_match or not y_match:
                print(f"Warning: Line skipped due to missing x/y values:\n{line}")
                continue
    
            try:
                x_coords = list(map(float, x_match[0].split()))
                y_coords = list(map(float, y_match[0].split()))
            except ValueError:
                print(f"Warning: Line has non-numeric coordinates:\n{line}")
                continue
    
            text_match = re.search(r"transcriptions:\s*\[u?[\"'](.+?)[\"']\]", line)
            text = text_match.group(1) if text_match else ""
    
            if len(x_coords) == len(y_coords) and len(x_coords) >= 4:
                polygon = list(zip(x_coords, y_coords))
                gts.append({'polygon': polygon, 'text': text})
            else:
                print(f"Warning: Polygon has mismatched or insufficient points:\n{line}")
    
        return gts
