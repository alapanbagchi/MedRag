# MedRAG Retrieval V2 Output

**Query:** Which surgical repair techniques were associated with recurrent coarctation and what are the percentages and p-values?

**Coverage:** 1/1 requirements covered - 1.0

## Requirements
- **H1** [COVERED] recurrent coarctation / comparative_numerical

## Selected Papers
| Rank | Paper | Score | Requirements | Reason |
|---|---|---|---|---|
| 1 | PMC11743609 | 0.99 | H1 | ranked |
| 2 | PMC11997790 | 0.852 | H1 | ranked |
| 3 | PMC12009809 | 0.8093 | H1 | ranked |
| 4 | PMC12014972 | 0.7938 | H1 | ranked |
| 5 | PMC11970464 | 0.7865 | H1 | ranked |
| 6 | PMC12028424 | 0.7738 | H1 | ranked |
| 7 | PMC12082946 | 0.747 | H1 | ranked |
| 8 | PMC12055418 | 0.6177 | H1 | ranked |
| 9 | PMC12028279 | 0.6155 | H1 | ranked |
| 10 | PMC11697624 | 0.6014 | H1 | ranked |
| 11 | PMC11955238 | 0.5616 | H1 | ranked |
| 12 | PMC11997764 | 0.5579 | H1 | ranked |

## Final Evidence Set
| # | Chunk | Paper | Type | Reqs | MedCPT | Penalty | Final |
|---|---|---|---|---|---|---|---|
| 1 | PMC11743609_T1_row_5 | PMC11743609 | table_row | H1 | 11.607 | 0.0 | 1.0 |
| 2 | PMC11997790_ivaf042-T2_row_0 | PMC11997790 | table_row | H1 | 8.907 | 0.0 | 0.874 |
| 3 | PMC11743609_T1_row_3 | PMC11743609 | table_row | H1 | 11.399 | 0.0 | 0.994 |
| 4 | PMC11743609_T1_row_6 | PMC11743609 | table_row | H1 | 11.098 | 0.0 | 0.986 |
| 5 | PMC11743609_T1_row_7 | PMC11743609 | table_row | H1 | 10.977 | 0.0 | 0.983 |
| 6 | PMC12028424_0 | PMC12028424 | paragraph | H1 | 0.022 | 0.0 | 0.629 |
| 7 | PMC12082946_18 | PMC12082946 | paragraph | - | 0.0 | 0.0 | 0.601 |
| 8 | PMC12014972_3 | PMC12014972 | paragraph | H1 | 0.386 | 0.0 | 0.593 |
| 9 | PMC11743609_T1_row_8 | PMC11743609 | table_row | H1 | 10.774 | 0.0 | 0.977 |
| 10 | PMC12009809_7 | PMC12009809 | paragraph | H1 | -0.628 | 0.0 | 0.544 |
| 11 | PMC11743609_T2_row_7 | PMC11743609 | table_row | H1 | 6.278 | 0.0 | 0.926 |
| 12 | PMC11743609_T2_row_8 | PMC11743609 | table_row | H1 | 6.168 | 0.0 | 0.923 |
| 13 | PMC11743609_T2_row_3 | PMC11743609 | table_row | H1 | 5.971 | 0.0 | 0.918 |
| 14 | PMC11743609_T2_row_6 | PMC11743609 | table_row | H1 | 5.549 | 0.0 | 0.906 |
| 15 | PMC11743609_T2_row_5 | PMC11743609 | table_row | H1 | 5.365 | 0.0 | 0.901 |
| 16 | PMC11743609_0 | PMC11743609 | paragraph | H1 | 11.091 | 0.0 | 0.898 |
| 17 | PMC11743609_T2_row_10 | PMC11743609 | table_row | H1 | 7.463 | 0.0 | 0.888 |
| 18 | PMC11743609_T2_row_2 | PMC11743609 | table_row | H1 | 7.103 | 0.0 | 0.878 |

## Evidence Text
### #1 - PMC11743609_T1_row_5 (PMC11743609, table_row)

*Section:* Results

*PageIndex nodes:* 0048
*Detected fields:* percentage

**Context** (scored evidence object):

Section: Results
Table: Table 1: Patient characteristics.
Headers: Variables | Median [IQR] or n (%)
Row: VSD (%) | 14 (50)
Footnotes: Data are presented as median [IQR] or n/N (%).
n, number of patients for given variable; N, total number of patients; n = 28; *, n = 27; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect.

**Chunk text** (provenance):

Row: VSD (%)
Median [IQR] or n (%): 14 (50)

---

### #2 - PMC11997790_ivaf042-T2_row_0 (PMC11997790, table_row)

*Section:* RESULTS > Surgical technique

*PageIndex nodes:* 0032
*Detected fields:* percentage

**Context** (scored evidence object):

Section: RESULTS > Surgical technique
Table: Table 2: Operative variables
Headers: Aortic arch reconstruction | n (%)
Row: Aortic arch roof-plasty plus coarctation repair | 30 (100)
Footnotes: PA: pulmonary artery.

**Chunk text** (provenance):

Row: Aortic arch roof-plasty plus coarctation repair
n (%): 30 (100)

---

### #3 - PMC11743609_T1_row_3 (PMC11743609, table_row)

*Section:* Results

*PageIndex nodes:* 0048
*Detected fields:* percentage

**Context** (scored evidence object):

Section: Results
Table: Table 1: Patient characteristics.
Headers: Variables | Median [IQR] or n (%)
Row: Female sex (%) | 14 (50)
Footnotes: Data are presented as median [IQR] or n/N (%).
n, number of patients for given variable; N, total number of patients; n = 28; *, n = 27; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect.

**Chunk text** (provenance):

Row: Female sex (%)
Median [IQR] or n (%): 14 (50)

---

### #4 - PMC11743609_T1_row_6 (PMC11743609, table_row)

*Section:* Results

*PageIndex nodes:* 0048
*Detected fields:* percentage

**Context** (scored evidence object):

Section: Results
Table: Table 1: Patient characteristics.
Headers: Variables | Median [IQR] or n (%)
Row: BAV (%) | 19 (68)
Footnotes: Data are presented as median [IQR] or n/N (%).
n, number of patients for given variable; N, total number of patients; n = 28; *, n = 27; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect.

**Chunk text** (provenance):

Row: BAV (%)
Median [IQR] or n (%): 19 (68)

---

### #5 - PMC11743609_T1_row_7 (PMC11743609, table_row)

*Section:* Results

*PageIndex nodes:* 0048
*Detected fields:* percentage

**Context** (scored evidence object):

Section: Results
Table: Table 1: Patient characteristics.
Headers: Variables | Median [IQR] or n (%)
Row: Aortic arch hypoplasia (%) | 20 (71)
Footnotes: Data are presented as median [IQR] or n/N (%).
n, number of patients for given variable; N, total number of patients; n = 28; *, n = 27; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect.

**Chunk text** (provenance):

Row: Aortic arch hypoplasia (%)
Median [IQR] or n (%): 20 (71)

---

### #6 - PMC12028424_0 (PMC12028424, paragraph)

*Section:* Abstract

*Detected fields:* percentage, p-value, confidence_interval, effect_estimate

Background: Coarctation of the aorta (CoA) represents 5% to 7% of all congenital heart diseases. Surgery and interventional methods offer great short-term results, but the occurrence of postoperative hypertension associated with cardiovascular and cerebral vascular disease increases mortality and morbidity in the long term. This study aims to investigate risk factors associated with postoperative hypertension in pediatric patients with early repair of isolated aortic coarctation. Subjects and Methods: A total of 41 patients with isolated aortic coarctation were included. The mean age was 35.3 ± 46.34 days. Early repair under one month was performed in 65.9% of patients. In all except two patients, end-to-end anastomosis was used. A follow-up at two years revealed an incidence of 58.5% of hypertension. Using logistic regression, preoperative renin plasma concentration above the upper normal level (46.1 μUI/mL) was independently associated with the occurrence of hypertension (OR = 2.49, 95% CI = 2.001–5.03, p = 0.001). Conclusion: Coarctation of the aorta is not just a simple mechanical obstruction of the aorta and should be seen and managed as a systemic disease. Abnormal preoperative renin concentrations were independently associated with the occurrence of HT at follow-up, suggesting that vascular dysfunction could play a role in hypertension development after successful CoA repair, negatively influencing the long-term prognostic of these patients.

---

### #7 - PMC12082946_18 (PMC12082946, paragraph)

*Section:* Results

*PageIndex nodes:* 0033
*Detected fields:* percentage, p-value, confidence_interval

Table 6 compares the demographics, anthropometrics, SC lesions, and interventions of surviving and deceased patients. Although the patient's weight was not significantly lower in the deceased group at presentation, the median Z-score did not differ. However, the patient's age, height, weight, and weight Z-score on the last follow-up were significantly lower in the deceased group. Such a pronounced difference seems expected due to the mortality in infancy for the three cases in the deceased category. Moreover, based on FS at the initial presentation, the proportion of cases with suppressed myocardial contractility was statistically significant in the deceased group (p = 0.03). Nevertheless, age at presentation, patient sex, lesions comprising the complex, presence of pulmonary hypertension, and percentage of cases that underwent interventions did not yield statistically significant differences between the two groups. However, a Kaplan–Meier survival analysis for patients who underwent intervention and those without intervention demonstrated a mean survival time for the intervention group of 90.2 months (SE = 4.7, 95% CI [80.9, 99.4]) and the non-intervention group 0.8 months (SE = 0.2, 95% CI [0.4, 1.1]) with log-rank test indicating a significant difference between the intervention and no-intervention survival distribution with a p-value of less than 0.001. Figure 4 illustrates the Kaplan–Meier survival curves. Fig. 4Cumulative Kaplan–Meier survival curves for SC patients with or without interventions (surgical/catheter). The p-value between patients'survival with or without interventions is < 0.001

---

### #8 - PMC12014972_3 (PMC12014972, paragraph)

*Section:* Background

*PageIndex nodes:* 0009
*Detected fields:* percentage

Coarctation of the aorta (CoA) accounts for 5–8% of all congenital heart defects. Transcatheter stent implantation has become the standard of care for native or recurrent coarctation [1]. Stent migration is a potential procedure complication observed in approximately 5% of cases [2]. Although proximal stent migration usually requires surgical intervention [3, 4], transcatheter management has been described in few cases [5, 6]. We describe a case of successful transcatheter management of this unusual complication.

---

### #9 - PMC11743609_T1_row_8 (PMC11743609, table_row)

*Section:* Results

*PageIndex nodes:* 0048
*Detected fields:* percentage

**Context** (scored evidence object):

Section: Results
Table: Table 1: Patient characteristics.
Headers: Variables | Median [IQR] or n (%)
Row: Bovine arch (%) | 4 (14)
Footnotes: Data are presented as median [IQR] or n/N (%).
n, number of patients for given variable; N, total number of patients; n = 28; *, n = 27; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect.

**Chunk text** (provenance):

Row: Bovine arch (%)
Median [IQR] or n (%): 4 (14)

---

### #10 - PMC12009809_7 (PMC12009809, paragraph)

*Section:* Treatment for AoC > Surgery

*Detected fields:* percentage

Complications, including left recurrent laryngeal nerve injury, bronchial compression, early re-coarctation, and paradoxical hypertension, occur in approximately 5% of patients. Older age at repair (>20 years) and preoperative hypertension are associated with decreased survival rates (10). Patients younger than 9 years at the time of repair showed significantly lower rates of hypertension at 5–15 years of follow-up. Additionally, younger age at repair and end-to-end anastomosis correction are linked to fewer reintervention on the descending aorta.

---

### #11 - PMC11743609_T2_row_7 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* percentage, p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: Preoperative arch hypoplasia (%) | 19 (79) | 1 (25) | 0.058
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: Preoperative arch hypoplasia (%)
No re-CoA (n = 24) [IQR] or n (%): 19 (79)
re-CoA (n = 4) [IQR] or n (%): 1 (25)
p: 0.058

---

### #12 - PMC11743609_T2_row_8 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* percentage, p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: Bovine arch (%) | 3 (13) | 1 (25) | 0.5
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: Bovine arch (%)
No re-CoA (n = 24) [IQR] or n (%): 3 (13)
re-CoA (n = 4) [IQR] or n (%): 1 (25)
p: 0.5

---

### #13 - PMC11743609_T2_row_3 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* percentage, p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: Female sex (%) | 12 (50) | 2 (50) | 0.7
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: Female sex (%)
No re-CoA (n = 24) [IQR] or n (%): 12 (50)
re-CoA (n = 4) [IQR] or n (%): 2 (50)
p: 0.7

---

### #14 - PMC11743609_T2_row_6 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* percentage, p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: VSD (%) | 12 (50) | 2 (50) | 1
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: VSD (%)
No re-CoA (n = 24) [IQR] or n (%): 12 (50)
re-CoA (n = 4) [IQR] or n (%): 2 (50)
p: 1

---

### #15 - PMC11743609_T2_row_5 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* percentage, p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: BAV (%) | 18 (75) | 1 (25) | 0.08
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: BAV (%)
No re-CoA (n = 24) [IQR] or n (%): 18 (75)
re-CoA (n = 4) [IQR] or n (%): 1 (25)
p: 0.08

---

### #16 - PMC11743609_0 (PMC11743609, paragraph)

*Section:* Abstract > Background

*Detected fields:* percentage

Recurrent coarctation of the aorta (re-CoA) is a well-known although not fully understood complication after surgical repair, typically occurring in 10%–20% of cases within months after discharge.

---

### #17 - PMC11743609_T2_row_10 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: Age at repair (days) | 9 [11] | 5.5 [6] | 0.12
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: Age at repair (days)
No re-CoA (n = 24) [IQR] or n (%): 9 [11]
re-CoA (n = 4) [IQR] or n (%): 5.5 [6]
p: 0.12

---

### #18 - PMC11743609_T2_row_2 (PMC11743609, table_row)

*Section:* Results > MRI assessment > Recurrent coarctation (re-CoA)

*PageIndex nodes:* 0060, 0058
*Detected fields:* p-value

**Context** (scored evidence object):

Section: Results > MRI assessment > Recurrent coarctation (re-CoA)
Table: Table 2: Characteristics of patients with and without early re-CoA.
Headers: Variables | No re-CoA (n = 24) [IQR] or n (%) | re-CoA (n = 4) [IQR] or n (%) | p
Row: Weight at birth (grams) | 3,337 [681] | 2,850 [953] | 0.12
Footnotes: Data are presented as median [IQR], n/N (%).
n, number of patients for given variable; N, total number of patients; AAo, ascending aorta; DAo, descending aorta; LCCA, left common carotid artery; LSA, left subclavian artery; S-to-D, systole-to-diastole; BAV, bicuspid aortic valve; re-CoA, recurrent coarctation; VSD, ventricular septal defect; No re-CoA: n = 24 (except for blood pressure gradient: n = 23); re-CoA, n = 4 (except for angle between the proximal arch and brachiocephalic artery n = 3).

**Chunk text** (provenance):

Row: Weight at birth (grams)
No re-CoA (n = 24) [IQR] or n (%): 3,337 [681]
re-CoA (n = 4) [IQR] or n (%): 2,850 [953]
p: 0.12

---

## Regression Metrics (spec section 41)
```json
{
  "per_query_paper_metrics": {
    "q1": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11743609",
        "PMC12028424",
        "PMC12082946",
        "PMC12028279",
        "PMC12009809",
        "PMC11702069",
        "PMC11997764",
        "PMC12087259",
        "PMC11749030",
        "PMC11818419"
      ]
    },
    "q2": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11743609",
        "PMC12009809",
        "PMC12014972",
        "PMC11997790",
        "PMC11970464",
        "PMC12055418",
        "PMC11885197",
        "PMC11955238",
        "PMC11697624",
        "PMC12063084"
      ]
    },
    "q3": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11743609",
        "PMC11997790",
        "PMC11970464",
        "PMC12009809",
        "PMC12055418",
        "PMC11697624",
        "PMC12014972",
        "PMC12028424",
        "PMC12034379",
        "PMC11914319"
      ]
    },
    "q4": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11743609",
        "PMC11997790",
        "PMC11970464",
        "PMC12009809",
        "PMC11854409",
        "PMC11702069",
        "PMC11955238",
        "PMC12063084",
        "PMC12014972",
        "PMC12028279"
      ]
    }
  },
  "final_paper_recall": 0.0,
  "known_papers_found": [],
  "known_papers_missing": [],
  "branch_papers_preserved_in_selection": true,
  "requirement_recall": 1.0,
  "coverage_fraction": 1.0,
  "hop_coverage": {
    "H1": {
      "covered": true
    }
  },
  "final_evidence_chunk_recall": {},
  "n_final_evidence": 18,
  "pageindex_evidence_count": 15,
  "table_context_evidence_count": 13,
  "pageindex_status": {
    "PMC11743609": "success",
    "PMC11997790": "success",
    "PMC12009809": "success",
    "PMC12014972": "success",
    "PMC11970464": "success",
    "PMC12028424": "success",
    "PMC12082946": "success",
    "PMC12055418": "success",
    "PMC12028279": "success"
  },
  "note": "Paper recall and MRR are computed against branch-level known papers (spec 40-41); requirement recall is the fraction of hops with evidence in the final set."
}
```