import json

def load_cuneiform_catalog(path='catalog.json'):
    """Load the cuneiform catalog from a JSON file and convert to dictionary format."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            catalog_list = json.load(f)
            
        # Convert list to dictionary using 'name' as key
        if isinstance(catalog_list, list):
            catalog_dict = {}
            for entry in catalog_list:
                if isinstance(entry, dict) and 'name' in entry:
                    name = entry['name']
                    catalog_dict[name] = entry
            return catalog_dict
        elif isinstance(catalog_list, dict):
            # Already in dictionary format
            return catalog_list
        else:
            print(f"❌ Unexpected catalog format in {path}")
            return {}
            
    except FileNotFoundError:
        print(f"❌ Catalog file not found at {path}")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON from {path}")
        return {}

def match_reference_with_catalog(reference_text, catalog):
    """
    Match a reference string against the catalog case-insensitively.

    Uses direct match, first-word heuristics and simple OCR-aware
    character substitutions to tolerate common OCR errors.
    """
    if not reference_text or not reference_text.strip():
        return None

    # Normalize reference text to lowercase once at the start
    clean_ref = reference_text.strip().lower()

    # Create a lowercase version of the catalog for efficient O(1) lookups
    # This avoids looping for direct matches.
    lower_catalog = {k.lower(): v for k, v in catalog.items()}

    # 1. Direct match (case-insensitive)
    if clean_ref in lower_catalog:
        return lower_catalog[clean_ref]

    # 2. First-word match (case-insensitive)
    first_word = clean_ref.split(':')[0].split(' ')[0].strip()
    if first_word in lower_catalog:
        return lower_catalog[first_word]

    # OCR substitutions are already lowercase, so no changes needed here
    ocr_corrections = {
        'u': ['n', 'm', 'w'], 'i': ['l', '1', '|'], 'g': ['q', '9', '6'],
        'n': ['u', 'm'], 'm': ['n', 'u'], 'q': ['g', '9'], 'l': ['i', '1', '|']
    }

    # Loop through the original catalog to perform more complex matches
    for key, value in catalog.items():
        lower_key = key.lower()

        # 3. OCR-aware similarity check (case-insensitive)
        if len(lower_key) == len(first_word):
            similar_chars = 0
            for i, char in enumerate(first_word):
                if i < len(lower_key):
                    # Compare lowercase characters directly
                    if char == lower_key[i]:
                        similar_chars += 1
                    # Check against the lowercase OCR corrections
                    elif char in ocr_corrections.get(lower_key[i], []):
                        similar_chars += 0.9
            
            if len(lower_key) > 0 and (similar_chars / len(lower_key)) >= 0.9:
                return value

        # 4. Partial prefix match (case-insensitive)
        if clean_ref.startswith(lower_key) or first_word.startswith(lower_key):
            return value

        # 5. Reverse substring match (case-insensitive)
        if lower_key in clean_ref or lower_key in first_word:
            return value

    return None
