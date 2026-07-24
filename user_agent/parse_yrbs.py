#!/usr/bin/env python3
"""
parse_yrbs.py

Parse the pipe-delimited 2023 national YRBS export (XXHq.txt) and compute
weighted prevalence rates + conditional (dependency-chain) rates directly
from the real data, instead of guessing multipliers.
"""
import pandas as pd
import numpy as np
import json
import os

def load_pipe_table(path):
    header_cols = None
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # separator lines are almost entirely dashes
            stripped = line.strip()
            if set(stripped) <= {"-"}:
                continue
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            # split on '|' produces leading/trailing empty strings because
            # each line starts and ends with '|'
            parts = [p for i, p in enumerate(parts) if not (i == 0 and p == "") and not (i == len(parts) - 1 and p == "")]
            if header_cols is None:
                header_cols = parts
            else:
                rows.append(parts)
    df = pd.DataFrame(rows, columns=header_cols)
    return df

def weighted_rate(sub, col, roi_values):
# YRBS uses complex survey sampling where different respondents represent different numbers of students. 
# A 12th grader's response might "count" more than a 9th grader's because of the sampling design.
# Accounts for the fact that some students represent more people than others. 
# A 12th grader might represent 50 students, while a 9th grader represents 30 students

#Survey Question	                        Column	ROI Values  Meaning
#"Did you seriously consider suicide?"	    q27	    [1]	        "Yes"
#"How many times did you attempt suicide?"	q29	    [2,3,4,5]	"1 or more times"

# Remove rows with missing data in either the response column or weight
    valid = sub.dropna(subset=[col, "weight"])
    if len(valid) == 0:
        return np.nan
    
    # Create boolean mask: True if response is in our ROI (Region of Interest)
    is_roi = valid[col].isin(roi_values)
    # Weighted numerator: sum of weights for ROI responses
    # Weighted denominator: sum of all weights
    return (is_roi * valid["weight"]).sum() / valid["weight"].sum()

def weighted_rate_by(df, col, roi_values, group_col, group_map):
# weighted rates for each subgroup (e.g., by sex, by grade) and returns them as a dictionary. 
    out = {}
    for val, label in group_map.items():
        sub = df[df[group_col] == val]
        out[label] = weighted_rate(sub, col, roi_values)
    return out

def conditional_pair(df, outcome_col, outcome_roi, condition_col, condition_roi):
# There is dependency between variables,
# If we sample independently:
# P(sadness) = 0.40
# P(ideation | no sadness) = 0.04  # Very low without sadness
# P(attempt  | no ideation) = 0.01  # Extremely low without ideation
# But if we just flip independent coins, we'd get unrealistic combinations:
# Students with suicide attempts but no sadness (almost never happens in reality)
# Students with suicide plans but no ideation (contradictory)
# Calculates conditional probability: "What's the rate of outcome X given that condition Y is true/false?"

    # Subset where condition is TRUE
    cond_true = df[df[condition_col].isin(condition_roi)]
    # Subset where condition is FALSE NOT MISSING
    cond_false = df[~df[condition_col].isin(condition_roi) & df[condition_col].notna()]
    # Calculate outcome rate among those with condition=True 
    r_true = weighted_rate(cond_true, outcome_col, outcome_roi)
    # Calculate outcome rate among those with condition=False 
    r_false = weighted_rate(cond_false, outcome_col, outcome_roi)
    # Calculate multipliers relative to overall population rate
    mult_true = r_true / weighted_rate(df, outcome_col, outcome_roi)
    mult_false = r_false / weighted_rate(df, outcome_col, outcome_roi)
    return r_true, r_false, mult_true, mult_false

def conditional_pair_by_sex(df, outcome_col, outcome_roi, condition_col, condition_roi,
                             sex_col="q2", sex_map=None):
    
# Same as conditional_pair, but computed separately within each sex subgroup.
# This matters because the dependency-chain conditional rates (e.g. P(ideation | sad)) 
# may not be identical across sexes.

# Returns {"overall": {...}, "female": {...}, "male": {...}}, where each
# value has the same shape as conditional_pair's output (as a dict).
# If a sex subgroup has too few rows for a stable estimate (n < min_n),
# that subgroup falls back to "overall" rather than reporting a noisy
# small-sample rate silently.
  
    if sex_map is None:
        sex_map = {1: "female", 2: "male"}

    def _pair_dict(sub):
        r_true, r_false, mult_true, mult_false = conditional_pair(
            sub, outcome_col, outcome_roi, condition_col, condition_roi
        )
        return {
            "rate_given_true": r_true,
            "rate_given_false": r_false,
            "multiplier_true": mult_true,
            "multiplier_false": mult_false,
        }

    out = {"overall": _pair_dict(df)}

    min_n = 100  # below this, a subgroup's conditional estimate is too noisy to trust on its own
    for sex_val, label in sex_map.items():
        sub = df[df[sex_col] == sex_val]
        # need enough rows in BOTH condition=true and condition=false branches,
        # not just enough rows overall
        n_cond_true = sub[condition_col].isin(condition_roi).sum()
        n_cond_false = (~sub[condition_col].isin(condition_roi) & sub[condition_col].notna()).sum()
        if n_cond_true < min_n or n_cond_false < min_n:
            print(f"  [warn] {label} subgroup too small for '{outcome_col} | {condition_col}' "
                  f"(n_true={n_cond_true}, n_false={n_cond_false}); falling back to overall")
            out[label] = dict(out["overall"])
            out[label]["_fallback_to_overall"] = True
        else:
            out[label] = _pair_dict(sub)

    return out

INPUT_TXT_PATH = r"C:\Users\lolin\Downloads\useragent\XXHq.txt"        
WORK_DIR = r"C:\Users\lolin\Downloads\useragent\yrbs_output"             
PICKLE_PATH = os.path.join(WORK_DIR, "yrbs_2023_parsed.pkl")
RESULTS_JSON_PATH = os.path.join(WORK_DIR, "real_conditional_rates.json")

def main():
    os.makedirs(WORK_DIR, exist_ok=True)
 
    df = load_pipe_table(INPUT_TXT_PATH)
    print("Parsed shape:", df.shape)
    print("Columns sample:", df.columns.tolist()[:15])

    # Convert relevant columns to numeric, blanks -> NaN
    cols_needed = ["q2", "q3", "q14", "q15", "q24", "q26", "q27", "q28", "q29",
               "q42", "q46", "q84", "weight"]
    for c in cols_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    print("\nNon-null counts:")
    print(df[cols_needed].notna().sum())

    df.to_pickle(PICKLE_PATH)
    print("\nSaved parsed dataframe to .pkl")

    df = pd.read_pickle(PICKLE_PATH)

    SEX_MAP = {1: "female", 2: "male"}

    # Sanity check against known published national numbers 
    print("=== SANITY CHECK vs published 2023 YRBS national figures ===")
    print(f"Sad/hopeless overall:      {weighted_rate(df, 'q26', [1]):.3f}   (published: 0.397)")
    print(f"Sad/hopeless by sex:       {weighted_rate_by(df, 'q26', [1], 'q2', SEX_MAP)}   (published: female~0.53, male~0.28)")
    print(f"Poor mental health overall:{weighted_rate(df, 'q84', [4,5]):.3f}   (published: 0.285)")
    print(f"Considered suicide overall:{weighted_rate(df, 'q27', [1]):.3f}   (published: 0.204)")
    print(f"Suicide plan overall:      {weighted_rate(df, 'q28', [1]):.3f}   (published: ~0.17)")
    print(f"Suicide attempt overall:   {weighted_rate(df, 'q29', [2,3,4,5]):.3f}   (published: 0.095)")
    print(f"Bullied at school overall: {weighted_rate(df, 'q24', [1]):.3f}   (published: 0.19)")
    print(f"Missed school (safety):    {weighted_rate(df, 'q14', [2,3,4,5]):.3f}   (published: 0.13)")
    print(f"Weapon threat at school:   {weighted_rate(df, 'q15', [2,3,4,5,6,7,8]):.3f}   (published: 0.09)")
    print(f"Ever marijuana overall:    {weighted_rate(df, 'q46', [2,3,4,5,6,7]):.3f}   (published: 0.295)")
    print(f"Current alcohol overall:   {weighted_rate(df, 'q42', [2,3,4,5,6,7]):.3f}   (published: ~0.223)")

    # REAL conditional rates for the dependency chain, broken out by sex 
    print("\n=== REAL conditional rates, by sex ===")

    pairs = [
        ("poor_mh given sad",      "q84", [4,5], "q26", [1]),
        ("ideation given sad",     "q27", [1],   "q26", [1]),
        ("plan given ideation",    "q28", [1],   "q27", [1]),
        ("attempt given plan",     "q29", [2,3,4,5], "q28", [1]),
        ("missed_school given bullied", "q14", [2,3,4,5], "q24", [1]),
    ]

    results = {}
    for name, outcome_col, outcome_roi, cond_col, cond_roi in pairs:
        print(f"\n{name}:")
        by_sex = conditional_pair_by_sex(df, outcome_col, outcome_roi, cond_col, cond_roi,
                                          sex_col="q2", sex_map=SEX_MAP)
        results[name] = by_sex
        for label, vals in by_sex.items():
            flag = " (fallback to overall)" if vals.get("_fallback_to_overall") else ""
            print(f"  {label:8s} rate|True: {vals['rate_given_true']:.3f}  rate|False: {vals['rate_given_false']:.3f}  "
                  f"-> mult_true={vals['multiplier_true']:.2f}, mult_false={vals['multiplier_false']:.2f}{flag}")
            
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_JSON_PATH}")
 
if __name__ == "__main__":
    main()