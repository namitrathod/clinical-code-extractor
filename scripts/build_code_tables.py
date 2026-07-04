"""Build the curated ICD-10 and CPT code whitelists for the v1 project scope.

Why a curated subset instead of the full 70k ICD-10 table?
- A model can only learn codes it sees enough examples of. 50 codes x ~80
  notes each is a learnable task; 70,000 codes with 0-2 examples each is not.
- Evaluation is only meaningful when every code has test coverage.
- The whitelist doubles as the inference-time validation filter: any code the
  model generates that is NOT in these files gets rejected as a hallucination.

Outputs (all in codes/):
- icd10_whitelist.json   flat {code: description} - used by eval + validation
- cpt_whitelist.json     flat {code: description} - used by eval + validation
- icd10_categories.json  {category: [codes]} - used by the synthetic data
                         generator in Milestone 2 to group conditions
"""

import json
import re
from pathlib import Path

# Descriptions match the official ICD-10-CM 2026 short descriptions.
# Grouped by clinical category so Milestone 2 can build varied condition
# profiles (a diabetes note looks nothing like a knee-pain note).
ICD10_BY_CATEGORY: dict[str, dict[str, str]] = {
    "endocrine_metabolic": {
        "E11.9":   "Type 2 diabetes mellitus without complications",
        "E11.65":  "Type 2 diabetes mellitus with hyperglycemia",
        "E11.40":  "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
        "E78.5":   "Hyperlipidemia, unspecified",
        "E78.00":  "Pure hypercholesterolemia, unspecified",
        "E03.9":   "Hypothyroidism, unspecified",
        "E66.9":   "Obesity, unspecified",
        "E55.9":   "Vitamin D deficiency, unspecified",
    },
    "cardiovascular": {
        "I10":     "Essential (primary) hypertension",
        "I25.10":  "Atherosclerotic heart disease of native coronary artery without angina pectoris",
        "I48.91":  "Unspecified atrial fibrillation",
        "I50.9":   "Heart failure, unspecified",
        "I73.9":   "Peripheral vascular disease, unspecified",
    },
    "respiratory": {
        "J06.9":   "Acute upper respiratory infection, unspecified",
        "J02.9":   "Acute pharyngitis, unspecified",
        "J01.90":  "Acute maxillary sinusitis, unspecified",
        "J20.9":   "Acute bronchitis, unspecified",
        "J45.909": "Unspecified asthma, uncomplicated",
        "J44.9":   "Chronic obstructive pulmonary disease, unspecified",
        "J44.1":   "Chronic obstructive pulmonary disease with (acute) exacerbation",
        "J18.9":   "Pneumonia, unspecified organism",
    },
    "musculoskeletal": {
        "M54.50":  "Low back pain, unspecified",
        "M54.2":   "Cervicalgia",
        "M25.561": "Pain in right knee",
        "M25.562": "Pain in left knee",
        "M17.11":  "Unilateral primary osteoarthritis, right knee",
        "M19.90":  "Unspecified osteoarthritis, unspecified site",
        "M79.10":  "Myalgia, unspecified site",
    },
    "mental_neuro": {
        "F41.1":   "Generalized anxiety disorder",
        "F41.9":   "Anxiety disorder, unspecified",
        "F32.9":   "Major depressive disorder, single episode, unspecified",
        "F33.1":   "Major depressive disorder, recurrent, moderate",
        "F17.210": "Nicotine dependence, cigarettes, uncomplicated",
        "G47.00":  "Insomnia, unspecified",
        "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
        "R51.9":   "Headache, unspecified",
    },
    "gastrointestinal": {
        "K21.9":   "Gastro-esophageal reflux disease without esophagitis",
        "K59.00":  "Constipation, unspecified",
        "K58.9":   "Irritable bowel syndrome without diarrhea",
        "K29.70":  "Gastritis, unspecified, without bleeding",
        "R10.9":   "Unspecified abdominal pain",
    },
    "genitourinary_renal": {
        "N39.0":   "Urinary tract infection, site not specified",
        "N18.30":  "Chronic kidney disease, stage 3 unspecified",
        "N40.0":   "Benign prostatic hyperplasia without lower urinary tract symptoms",
        "D64.9":   "Anemia, unspecified",
    },
    "infectious_skin": {
        "B34.9":   "Viral infection, unspecified",
        "J11.1":   "Influenza due to unidentified influenza virus with other respiratory manifestations",
        "L03.90":  "Cellulitis, unspecified",
        "H66.90":  "Otitis media, unspecified, unspecified ear",
    },
    "wellness_symptoms": {
        "Z00.00":  "Encounter for general adult medical examination without abnormal findings",
        "Z23":     "Encounter for immunization",
        "Z12.11":  "Encounter for screening for malignant neoplasm of colon",
        "R05.9":   "Cough, unspecified",
        "R50.9":   "Fever, unspecified",
        "R53.83":  "Other fatigue",
        "R42":     "Dizziness and giddiness",
    },
}

# CPT codes. E/M (evaluation & management) codes are the office-visit levels;
# the rest are common in-office procedures and labs that our synthetic notes
# can plausibly mention ("labs drawn today", "EKG performed", etc.).
CPT_CODES: dict[str, str] = {
    # Office visits - established patient (level = complexity/time)
    "99212": "Office/outpatient visit, established patient, straightforward complexity",
    "99213": "Office/outpatient visit, established patient, low complexity",
    "99214": "Office/outpatient visit, established patient, moderate complexity",
    "99215": "Office/outpatient visit, established patient, high complexity",
    # Office visits - new patient
    "99202": "Office/outpatient visit, new patient, straightforward complexity",
    "99203": "Office/outpatient visit, new patient, low complexity",
    "99204": "Office/outpatient visit, new patient, moderate complexity",
    "99205": "Office/outpatient visit, new patient, high complexity",
    # Preventive medicine (annual physicals), by age band
    "99385": "Preventive visit, new patient, age 18-39",
    "99386": "Preventive visit, new patient, age 40-64",
    "99395": "Preventive visit, established patient, age 18-39",
    "99396": "Preventive visit, established patient, age 40-64",
    # Common in-office procedures and point-of-care tests
    "36415": "Collection of venous blood by venipuncture",
    "93000": "Electrocardiogram, routine ECG with at least 12 leads, with interpretation and report",
    "71046": "Radiologic examination, chest, 2 views",
    "81002": "Urinalysis, non-automated, without microscopy",
    "87880": "Infectious agent antigen detection, Streptococcus group A, rapid test",
    "90471": "Immunization administration, one vaccine",
    "96127": "Brief emotional/behavioral assessment, standardized instrument",
    "20610": "Arthrocentesis, aspiration and/or injection, major joint or bursa",
    # Common lab panels (billed with the venipuncture)
    "80053": "Comprehensive metabolic panel",
    "85025": "Complete blood count with automated differential",
    "83036": "Hemoglobin A1c level",
    "94010": "Spirometry, including graphic record",
}

# Official ICD-10-CM structure: letter + 2 digits, then optional . + 1-4
# alphanumerics (e.g. I10, E11.65, M25.561, F17.210, Z00.00).
ICD10_PATTERN = re.compile(r"^[A-TV-Z]\d{2}(?:\.[0-9A-Z]{1,4})?$")
# CPT codes are exactly 5 digits.
CPT_PATTERN = re.compile(r"^\d{5}$")


def validate(codes: dict[str, str], pattern: re.Pattern, kind: str) -> None:
    """Fail loudly if any curated code doesn't match the official format.

    This catches typos at build time instead of letting them silently poison
    the dataset and every downstream metric.
    """
    bad = [c for c in codes if not pattern.match(c)]
    if bad:
        raise ValueError(f"Malformed {kind} codes: {bad}")
    empty = [c for c, desc in codes.items() if not desc.strip()]
    if empty:
        raise ValueError(f"{kind} codes missing descriptions: {empty}")


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "codes"
    out_dir.mkdir(exist_ok=True)

    # Flatten {category: {code: desc}} -> {code: desc}
    icd10_flat: dict[str, str] = {}
    for category, codes in ICD10_BY_CATEGORY.items():
        overlap = icd10_flat.keys() & codes.keys()
        if overlap:
            raise ValueError(f"Duplicate ICD-10 codes across categories: {overlap}")
        icd10_flat.update(codes)

    validate(icd10_flat, ICD10_PATTERN, "ICD-10")
    validate(CPT_CODES, CPT_PATTERN, "CPT")

    categories = {cat: sorted(codes) for cat, codes in ICD10_BY_CATEGORY.items()}

    (out_dir / "icd10_whitelist.json").write_text(
        json.dumps(icd10_flat, indent=2), encoding="utf-8")
    (out_dir / "cpt_whitelist.json").write_text(
        json.dumps(CPT_CODES, indent=2), encoding="utf-8")
    (out_dir / "icd10_categories.json").write_text(
        json.dumps(categories, indent=2), encoding="utf-8")

    print(f"ICD-10 whitelist: {len(icd10_flat)} codes "
          f"across {len(ICD10_BY_CATEGORY)} categories")
    for cat, codes in ICD10_BY_CATEGORY.items():
        print(f"  {cat:24s} {len(codes)} codes")
    print(f"CPT whitelist:    {len(CPT_CODES)} codes")
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
