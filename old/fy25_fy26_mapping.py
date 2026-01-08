"""
FY25 to FY26 Question Mapping
This file contains the mapping between FY25 and FY26 question IDs and answer options
for tracking changes across survey years.

FY25 Data Structure:
- Responses stored in Responses_FY25Q3.csv and Responses_FY25Q4.csv
- Long format with columns: QuestionID, Question, Answer, ExtractedAnswer
- Products Heard: Q32
- Techniques Heard: Q32.1
- Products Used: Q33
- Techniques Used: Q33.1

FY26 Data Structure:
- Responses stored in CLEAN_FY26.csv
- Wide format with columns for each question option
- Products Heard: Q31_a, Q31_b, etc.
- Techniques Heard: Q32_a, Q32_b, etc.
- Products Used: Q33_a, Q33_b, etc.
- Techniques Used: Q34_a, Q34_b, etc.
"""

import pandas as pd
import csv

# Mapping: FY26_Column -> (FY25_QuestionID, FY25_Answer_Text, Label)
FY25_TO_FY26_MAPPING = {
    # Yetagon Products/Services - Heard Of (FY26 Q31 -> FY25 Q32)
    'Q31_a': ('Q32', 'Yetagon Irrigation', 'Yetagon Irrigation (ရေတံခွန် ရေသွင်းပစ္စည်း)'),
    'Q31_b': ('Q32', 'Zarmani', 'Yetagon EM/Zarmani (ရေတံခွန် ဇာမဏီစွမ်း)'),
    'Q31_c': ('Q32', 'Barhmati', 'Yetagon Trichoderma/Barhmati (ရေတံခွန် ဘာမထီစမ်း)'),
    'Q31_d': ('Q32', 'Yetagon Tele-agronomy', 'Yetagon Tele-agronomy'),
    'Q31_e': ('Q32', None, 'Yetagon Sun-kissed (နေခြည်ဆမ်း)'),  # Not in FY25
    'Q31_f': ('Q32', None, 'Yetagon Fish Amino (ပုလဲဆန်း)'),  # Not in FY25
    'Q31_g': ('Q34', None, 'Po Chat'),  # Changed question in FY25
    'Q31_h': ('Q32', None, 'Messenger'),  # Not in FY25
    'Q31_i': ('Q32', 'Digital farm practices', 'Digital farm practices'),
    'Q31_j': ('Q32', 'None of the above', 'None of the above'),
    'Q31_k': ('Q32', 'Zarmani', 'EM'),  # Same as Q31_b

    # FY25 Q32 Only - Not in FY26
    'NA': ('Q32', 'Yetagon On-call Service', 'Yetagon On-call Service'),
    'NA': ('Q32', 'Bio Products (EM or Liquid Trichoderma)', 'Bio Products (EM or Liquid Trichoderma)'),
    'NA': ('Q32', 'Pest and disease prevention services (သီးနှံကာကွယ်ရေးအကြံပေး)', 'Pest and disease prevention services'),
    'NA': ('Q32', 'Yetagon Tele-agronomy and digital farm practices', 'Yetagon Tele-agronomy and digital farm practices'),

    # Yetagon Products/Services - Used (FY26 Q33 -> FY25 Q33)
    'Q33_a': ('Q33', 'Yetagon Irrigation', 'Yetagon Irrigation (ရေတံခွန် ရေသွင်းပစ္စည်း)'),
    'Q33_b': ('Q33', 'Zarmani', 'Yetagon EM/Zarmani (ရေတံခွန် ဇာမဏီစွမ်း)'),
    'Q33_c': ('Q33', 'Barhmati', 'Yetagon Trichoderma/Barhmati (ရေတံခွန် ဘာမထီစမ်း)'),
    'Q33_d': ('Q33', 'Yetagon Tele-agronomy', 'Yetagon Tele-agronomy'),
    'Q33_i': ('Q33', None, 'Yetagon Sun-kissed (နေခြည်ဆမ်း)'),  # Not in FY25
    'Q33_j': ('Q33', None, 'Yetagon Fish Amino (ပုလဲဆန်း)'),  # Not in FY25
    'Q33_k': ('Q33', None, 'Po Chat'),  # Not in FY25
    'Q33_l': ('Q33', None, 'Messenger'),  # Not in FY25
    'Q33_f': ('Q33', 'Digital farm practices', 'Digital farm practices'),

    # FY25 Q33 Only - Not in FY26
    'NA': ('Q33', 'Yetagon On-call Service', 'Yetagon On-call Service'),
    'NA': ('Q33', 'Bio Products (EM or Liquid Trichoderma)', 'Bio Products (EM or Liquid Trichoderma)'),
    'NA': ('Q33', 'Pest and disease prevention services (သီးနှံကာကွယ်ရေးအကြံပေး)', 'Pest and disease prevention services'),
    'NA': ('Q33', 'Yetagon Tele-agronomy and digital farm practices', 'Yetagon Tele-agronomy and digital farm practices'),

    # Farming Techniques - Heard Of (FY26 Q32 -> FY25 Q32.1)
    'Q32_a': ('Q32.1', 'No Burn Rice Farming', 'No Burn Rice Farming'),
    'Q32_b': ('Q32.1', 'Salt Water Seed Selection', 'Salt Water Seed Selection'),
    'Q32_c': ('Q32.1', 'Basal Fertilizer Usage for Rice', 'Basal Fertilizer Usage for Rice'),
    'Q32_d': ('Q32.1', 'Mid-season Fertilizer Usage for Rice', 'Mid-season Fertilizer Usage for Rice'),
    'Q32_e': ('Q32.1', 'Paddy Liming Acid', 'Paddy Liming Acid'),
    'Q32_f': ('Q32.1', 'Gypsum Appication', 'Gypsum Application'),
    'Q32_g': ('Q32.1', 'Boron Foliar Spray', 'Boron Foliar Spray'),
    'Q32_h': ('Q32.1', 'Epsom Salt Foliar Spray', 'Epsom Salt Foliar Spray'),
    'Q32_i': ('Q32.1', 'Neem Pesticide', 'Neem Pesticide'),
    'Q32_j': ('Q32.1', 'Fish Amino', 'Fish Amino'),

    # Farming Techniques - Used (FY26 Q34 -> FY25 Q33.1)
    'Q34_a': ('Q33.1', 'No Burn Rice Farming', 'No Burn Rice Farming'),
    'Q34_b': ('Q33.1', 'Salt Water Seed Selection', 'Salt Water Seed Selection'),
    'Q34_c': ('Q33.1', 'Basal Fertilizer Usage for Rice', 'Basal Fertilizer Usage for Rice'),
    'Q34_d': ('Q33.1', 'Mid-season Fertilizer Usage for Rice', 'Mid-season Fertilizer Usage for Rice'),
    'Q34_e': ('Q33.1', 'Paddy Liming Acid', 'Paddy Liming Acid'),
    'Q34_f': ('Q33.1', 'Gypsum Appication', 'Gypsum Application'),
    'Q34_g': ('Q33.1', 'Boron Foliar Spray', 'Boron Foliar Spray'),
    'Q34_h': ('Q33.1', 'Epsom Salt Foliar Spray', 'Epsom Salt Foliar Spray'),
    'Q34_i': ('Q33.1', 'Neem Pesticide', 'Neem Pesticide'),
    'Q34_j': ('Q33.1', 'Fish Amino', 'Fish Amino'),
}


def create_mapping_csv(output_path='FY25_to_FY26.csv'):
    """
    Create a CSV file with the FY25 to FY26 mapping

    Args:
        output_path: Path where the CSV file will be saved
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['FY26_Column', 'FY25_QuestionID', 'FY25_Answer_Text', 'Label', 'Notes'])

        for fy26_col, (fy25_qid, fy25_answer, label) in sorted(FY25_TO_FY26_MAPPING.items(), key=lambda x: (x[1][0], x[0])):
            notes = ''
            if fy25_answer is None:
                notes = 'Not in FY25'
            elif fy26_col == '':
                notes = 'Not in FY26'
            writer.writerow([fy26_col, fy25_qid, fy25_answer or '', label, notes])

    print(f"Mapping file created: {output_path}")
    return output_path


def load_fy25_responses(q3_path='Responses_FY25Q3.csv', q4_path='Responses_FY25Q4.csv'):
    """
    Load and combine FY25 Q3 and Q4 response data

    Returns:
        DataFrame with combined FY25 responses
    """
    df_q3 = pd.read_csv(q3_path)
    df_q4 = pd.read_csv(q4_path)

    # Combine both quarters
    df_combined = pd.concat([df_q3, df_q4], ignore_index=True)

    return df_combined


def transform_fy25_to_wide_format(fy25_responses, mapping=FY25_TO_FY26_MAPPING):
    """
    Transform FY25 long-format responses to FY26-compatible wide format

    Args:
        fy25_responses: DataFrame from load_fy25_responses()
        mapping: Mapping dictionary

    Returns:
        DataFrame in wide format matching FY26 structure
    """
    # Get unique farmers
    farmers = fy25_responses['Farmer\'s name'].unique()

    # Initialize result dataframe
    result = pd.DataFrame({'Farmer\'s name': farmers})

    # For each FY26 column in mapping
    for fy26_col, (fy25_qid, fy25_answer, label) in mapping.items():
        if fy26_col == '':
            # Skip empty column names (FY25-only items)
            continue

        result[fy26_col] = 0.0

        if fy25_answer is None:
            # Not available in FY25, keep as 0
            continue

        # Find matching responses using exact match on the answer portion after '/'
        # This handles both "Zarmani" and "Zarmani (ရေတံခွန် ဇာမဏီစွမ်း)" formats
        q_data = fy25_responses[fy25_responses['QuestionID'] == fy25_qid].copy()

        # Extract answer text from Question column (part after last '/')
        q_data['ExtractedAnswerText'] = q_data['Question'].str.split('/').str[-1]

        # Match based on answer text (check if it starts with the expected text)
        matching_mask = q_data['ExtractedAnswerText'].str.startswith(fy25_answer, na=False)
        matching_data = q_data[matching_mask]

        # For each farmer with Answer='1' (string), set the column to 1
        # Note: Answer column is object type containing string '1', not integer
        farmers_with_answer = matching_data[matching_data['Answer'] == '1']['Farmer\'s name'].unique()
        result.loc[result['Farmer\'s name'].isin(farmers_with_answer), fy26_col] = 1.0

    return result


def extract_all_fy25_answers(q3_path='Responses_FY25Q3.csv', q4_path='Responses_FY25Q4.csv'):
    """
    Extract all unique answers from FY25 quarterly response data

    Returns:
        Dictionary with QuestionID as key and list of unique answers as value
    """
    # Load FY25 quarterly data
    df_q3 = pd.read_csv(q3_path)
    df_q4 = pd.read_csv(q4_path)
    df_fy25 = pd.concat([df_q3, df_q4], ignore_index=True)

    result = {}

    # Extract Q32 (Products Heard)
    q32_questions = df_fy25[df_fy25['QuestionID'] == 'Q32']['Question'].unique()
    q32_answers = []
    for q in q32_questions:
        if '/' in q and 'None of the above' not in q and 'If other' not in q:
            answer = q.split('/')[-1]
            q32_answers.append(answer)
    result['Q32'] = sorted(q32_answers)

    # Extract Q32.1 (Techniques Heard)
    q321_questions = df_fy25[df_fy25['QuestionID'] == 'Q32.1']['Question'].unique()
    q321_answers = []
    for q in q321_questions:
        if '/' in q:
            answer = q.split('/')[-1]
            q321_answers.append(answer)
    result['Q32.1'] = sorted(q321_answers)

    # Extract Q33 (Products Used)
    q33_questions = df_fy25[df_fy25['QuestionID'] == 'Q33']['Question'].unique()
    q33_answers = []
    for q in q33_questions:
        if '/' in q and 'None of the above' not in q and 'If other' not in q:
            answer = q.split('/')[-1]
            q33_answers.append(answer)
    result['Q33'] = sorted(q33_answers)

    # Extract Q33.1 (Techniques Used)
    q331_questions = df_fy25[df_fy25['QuestionID'] == 'Q33.1']['Question'].unique()
    q331_answers = []
    for q in q331_questions:
        if '/' in q:
            answer = q.split('/')[-1]
            q331_answers.append(answer)
    result['Q33.1'] = sorted(q331_answers)

    return result


def generate_complete_mapping():
    """
    Programmatically generate complete FY25-FY26 mapping by extracting all answers from FY25 data
    """
    print("Extracting all unique answers from FY25 quarterly data...")
    fy25_answers = extract_all_fy25_answers()

    print("\n=== FY25 Q32 (Products Heard) ===")
    for ans in fy25_answers['Q32']:
        print(f"  '{ans}'")

    print("\n=== FY25 Q32.1 (Techniques Heard) ===")
    for ans in fy25_answers['Q32.1']:
        print(f"  '{ans}'")

    print("\n=== FY25 Q33 (Products Used) ===")
    for ans in fy25_answers['Q33']:
        print(f"  '{ans}'")

    print("\n=== FY25 Q33.1 (Techniques Used) ===")
    for ans in fy25_answers['Q33.1']:
        print(f"  '{ans}'")

    return fy25_answers


if __name__ == '__main__':
    # Generate complete mapping from FY25 data
    fy25_answers = generate_complete_mapping()

    # Create the mapping CSV file
    create_mapping_csv()

    print("\n=== Mapping Summary ===")
    print(f"Total FY26 columns mapped: {len(FY25_TO_FY26_MAPPING)}")

    not_in_fy25 = sum(1 for _, (_, answer, _) in FY25_TO_FY26_MAPPING.items() if answer is None)
    print(f"New in FY26 (not in FY25): {not_in_fy25}")
    print(f"Available in both years: {len(FY25_TO_FY26_MAPPING) - not_in_fy25}")
