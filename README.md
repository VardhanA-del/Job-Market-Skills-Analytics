# Job Market & Skills Analytics Dashboard Using Python and Streamlit

A portfolio-ready analytics project that turns a local 2025 Indian job-listing workbook into a reusable preprocessing pipeline, exploratory notebook, and six-view interactive Streamlit dashboard.

## Problem statement

Job seekers and analysts need a defensible way to compare demand for roles, skills, locations, companies, experience levels, and disclosed salaries. Raw job listings contain duplicate IDs, free-text titles and skills, multi-location strings, relative dates, and inconsistent or missing salary labels. This project cleans those fields without fabricating values and exposes the results through interactive, sample-aware analysis.

## Objectives

- Identify high-demand role groups, skills, companies, and locations.
- Measure skill co-occurrence and role-specific technology demand.
- Compare disclosed salaries by role, skill, experience, location, and company.
- Surface dataset-derived career and entry-level skill priorities.
- Keep processing logic reusable, transparent, and separate from presentation.

## Dataset

The source is the locally supplied `dataset/indian-job-market-dataset-2025.xlsx` workbook (`Sheet1`): 97,929 raw rows and 17 columns. It includes title, stable job ID, relative upload age, company, tags/skills, experience range, salary label, multi-location text, company rating/review metadata, description, and pre-extracted numeric ranges.

The workbook name suggests a 2025 Indian job-market dataset and the task context indicates it was already downloaded from Kaggle. The workbook itself does not contain a canonical source URL, platform-coverage statement, scrape timestamp, or data dictionary, so those details are not invented here.

### Limitations

- This is one supplied source snapshot and does not represent the entire Indian labor market.
- 64,076 raw rows say “Not disclosed”; salary findings apply only to usable disclosed values.
- 129 USD rows are retained in native units but excluded from INR/LPA comparisons because no dated exchange rate is available.
- Upload values are relative labels such as “4 Days Ago”; no absolute posting dates are fabricated.
- Role groups use documented regex rules and inevitably simplify varied titles.
- Salary, skill, and company patterns are observational associations, not causal effects.

## Architecture

```text
.
├── dataset/                         # Original workbook; never modified
├── data/processed/                  # Generated Parquet analytics tables + audit report
├── notebooks/exploratory_analysis.ipynb
├── pages/
│   ├── 1_Job_Market_Overview.py
│   ├── 2_Skills_Analytics.py
│   ├── 3_Salary_Analytics.py
│   ├── 4_Location_Analytics.py
│   ├── 5_Company_Analytics.py
│   └── 6_Career_Insights.py
├── scripts/
│   ├── build_processed_data.py
│   ├── create_notebook.py
│   └── inspect_dataset.py
├── src/
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── skills_processing.py
│   ├── metrics.py
│   ├── salary_analysis.py
│   ├── dashboard.py
│   └── utils.py
├── tests/test_pipeline.py
├── app.py
└── requirements.txt
```

## Technology stack

Python, Streamlit, Pandas, NumPy, Plotly, OpenPyXL, PyArrow, and Jupyter. No Kaggle API or credentials are used. Scikit-learn is intentionally omitted because the project does not ship a salary-prediction model.

## Data-cleaning methodology

- Column labels are converted from camelCase/punctuation to snake_case.
- The highest-completeness row is retained for each stable `jobId`; 250 duplicate-ID rows are removed, including 247 exact duplicates.
- Original title and location values remain available alongside normalized role and location fields.
- Company text is trimmed; four missing company names remain missing rather than being fabricated.
- Simple HTML is removed from descriptions while readable text is retained.
- Ratings outside 1–5 and negative review counts would become missing.
- Locations are split on commas and conservatively normalize common aliases (for example Bangalore → Bengaluru and Gurgaon → Gurugram). Multi-location jobs are stored in a separate unique bridge table.
- Experience uses numeric source fields where available, with text parsing as fallback. Strict Fresher means minimum 0 and maximum ≤2 years; Entry Level means minimum ≤2; Mid Level means minimum 3–5; Senior means minimum >5.

## Salary normalization

Salary text is parsed instead of treating missing values as zero:

- Lac/Lakh/LPA × 100,000.
- Crore/Cr × 10,000,000.
- Explicit monthly values × 12.
- Mixed ranges such as `70,000-2 Lacs PA` preserve both units correctly.
- `Not disclosed`, `Unpaid`, zero-only, or uninterpretable labels become missing.
- `minimum_salary`, `maximum_salary`, and `average_salary` retain native-currency annual values.
- LPA fields exist only for INR listings; USD remains currency-separated.

Median salary is emphasized and grouped comparisons require minimum sample sizes, displayed in chart titles and hover details. Distribution charts cap only the display at the 99th percentile; source rows are retained.

## Skills processing

The pipeline splits tags on comma, semicolon, pipe, and newline; removes repeat skills within a listing; and conservatively canonicalizes variants such as `python3` → Python, `powerbi` → Power BI, `microsoft excel` → Excel, `postgres` → PostgreSQL, and `amazon web services` → AWS. Important core technologies are also detected in descriptions. All actual tag skills remain discoverable; analysis is not restricted to a fixed vocabulary.

The job–skill bridge contains one row per unique job/skill, so demand and pair counts are not inflated. Co-occurrence uses unordered pairs among frequent skills.

## Dashboard views

1. **Job Market Overview** — KPIs, roles, locations, hiring companies, experience, salary distribution, and relative posting age.
2. **Skills Analytics** — skill demand, role/experience/location matrices, co-occurrence, salary associations, and a Data Analyst-specific comparison.
3. **Salary Analytics** — distribution, role box plots, experience relationship, and sample-aware role/location/skill/company medians.
4. **Location Analytics** — job volume, salary, skills, companies, experience, and a documented opportunity score.
5. **Company Analytics** — hiring volume, role/skill matrices, ratings, salary, and experience demand.
6. **Career Insights** — target-role profile, data-derived missing-skill recommender, and strict-fresher/broader-entry-level analysis.

Global sidebar filters cover supported role, primary location, experience, company, skill, salary, and rating fields. Empty selections fail gracefully.

## Core KPIs

Unique job listings, companies, primary locations, normalized role groups, median disclosed INR salary, median experience, most in-demand skill, and most in-demand role.

## Reproducible findings from this snapshot

- The clean table has 97,679 unique job IDs; 97,186 listings have at least one extracted skill.
- Software Development is the largest named role group (13,650 listings); “Other” remains larger because conservative rules avoid forcing unrelated titles into a category.
- The leading extracted skills include Sales (9,178), Python (7,558), SQL (7,171), Project Management (5,544), and Customer Service (5,122).
- Bengaluru is the most common normalized primary location (19,426), followed by Hyderabad (12,333) and Pune (9,378).
- Accenture has the most listings in the snapshot (8,193), followed by IDESLABS (2,294) and Wipro (1,494).
- Among 33,362 usable disclosed INR listings, the median average range midpoint is ₹4.2 LPA. This is not a market-wide salary estimate.
- Data Engineer has 1,021 listings and a ₹19.6 LPA median among its 310 disclosed-salary records; Data Analyst has 301 listings and ₹7.8 LPA among 61 records. Small-sample safeguards remain essential.

## Installation and use

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python scripts/build_processed_data.py
streamlit run app.py
```

The second command is optional: the app builds processed files automatically if they are absent. It never overwrites the raw workbook.

Run validation:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src pages scripts app.py
```

Regenerate the notebook structure if needed:

```bash
python scripts/create_notebook.py
```

### Run the notebook in Google Colab

1. Upload `notebooks/exploratory_analysis.ipynb` to Colab.
2. Upload the dataset workbook directly into Colab's `/content` folder.
3. Choose **Runtime → Run all**.

The notebook is self-contained and does not import the local `src` package. It discovers the uploaded filename automatically, installs only missing libraries, and writes generated outputs under `/content/job_market_outputs` without modifying the uploaded dataset.

## Screenshots

Add portfolio screenshots here after running the app locally:

- Overview dashboard
- Skills and co-occurrence analysis
- Salary comparison
- Career Insights recommender

## Future improvements

- Add an authoritative scrape timestamp and source data dictionary.
- Track repeated snapshots to support real calendar trends.
- Add state/region mapping from a validated geographic reference table.
- Evaluate salary prediction only after stronger coverage, currency context, and outlier labeling are available.
- Add user-selectable normalization review tables for ambiguous titles and skills.
