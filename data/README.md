# Data Directory & EDA Documentation

- `raw/`: Unmodified raw dataset files.
  - `mental_health_dataset.csv`: 3,553 records, 116 columns.
- `interim/`: Intermediate transformed and cleaned dataset files (Step 4).
- `processed/`: Splitted train/validation/test datasets (Step 5+).
- `external/`: External lexicons and pre-trained vector embeddings.

## Key EDA Observations (Raw Dataset)
- **Total Records**: 3,553
- **Primary Text Column**: `text`
- **Primary Label Column**: `label` (Binary: `0` = Non-Stress [47.73%], `1` = Stress [52.27%])
- **Class Balance**: Balanced dataset (Majority/Minority ratio = 1.09)
- **Missing Values**: 0 missing values across all 116 columns (100% complete)
- **Text Length Statistics**:
  - Mean Word Count: 84.12 words
  - Median Word Count: 78.00 words
  - Min/Max Word Count: 3 to 284 words
- **Duplicates & Data Leakage Risk**:
  - 21 exact text duplicates across rows.
  - 624 recurring `post_id` entries due to multi-sentence segmentation.
  - Stratified text-level deduplication is recommended prior to train/val/test split.
