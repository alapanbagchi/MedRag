# MedRAG Retrieval V2 Output

**Query:** Which surgical repair techniques were associated with recurrent coarctation and what are the percentages and p-values?

**Coverage:** 4/4 requirements covered - 1.0

## Requirements
- **H1** [COVERED] Specific surgical repair techniques used for coarctation. / evidence
- **H2** [COVERED] Reported recurrence rates for each technique or comparison. / numerical
- **H3** [COVERED] Statistical significance (p-values) associated with recurrence for each technique or comparison. / numerical
- **H4** [COVERED] Percentages of recurrence for each technique or comparison. / numerical

## Selected Papers
| Rank | Paper | Score | Requirements | Reason |
|---|---|---|---|---|
| 1 | PMC11743609 | 1.1274 | H1, H2, H3, H4 | ranked |
| 2 | PMC12009809 | 1.0019 | H1, H2, H3, H4 | ranked |
| 3 | PMC11970464 | 0.9569 | H1, H2, H3, H4 | ranked |
| 4 | PMC11997790 | 0.9183 | H1, H2, H4 | ranked |
| 5 | PMC11810876 | 0.8691 | H1, H2, H3, H4 | ranked |
| 6 | PMC12028424 | 0.8311 | H1, H2, H3 | ranked |
| 7 | PMC11970461 | 0.7497 | H1, H2 | ranked |
| 8 | PMC11787011 | 0.7451 | H1, H2, H3 | ranked |
| 9 | PMC12082946 | 0.7132 | H1, H3, H4 | ranked |
| 10 | PMC11854409 | 0.7033 | H1, H2 | ranked |
| 11 | PMC11697624 | 0.6915 | H1, H2, H3 | ranked |
| 12 | PMC11914319 | 0.6804 | H1, H3, H4 | ranked |

## Final Evidence Set
| # | Chunk | Paper | Type | Reqs | MedCPT | Penalty | Final |
|---|---|---|---|---|---|---|---|
| 1 | PMC12009809_6 | PMC12009809 | paragraph | H1, H2, H3, H4 | 0.032 | 0.0 | 1.0 |
| 2 | PMC12028424_11 | PMC12028424 | paragraph | H1, H3 | 0.03 | 0.0 | 0.982 |
| 3 | PMC11743609_0 | PMC11743609 | paragraph | - | 0.032 | 0.0 | 0.966 |
| 4 | PMC12082946_17 | PMC12082946 | paragraph | - | 0.033 | 0.0 | 0.966 |
| 5 | PMC11970461_18 | PMC11970461 | paragraph | - | 0.032 | 0.0 | 0.966 |
| 6 | PMC11787011_5 | PMC11787011 | paragraph | - | 0.033 | 0.0 | 0.966 |
| 7 | PMC11697624_3 | PMC11697624 | paragraph | - | 0.033 | 0.0 | 0.961 |
| 8 | PMC11914319_10 | PMC11914319 | paragraph | - | 0.033 | 0.0 | 0.957 |
| 9 | PMC11997790_2 | PMC11997790 | paragraph | - | 0.03 | 0.0 | 0.943 |
| 10 | PMC12014972_3 | PMC12014972 | paragraph | - | 0.033 | 0.0 | 0.942 |
| 11 | PMC12009809_7 | PMC12009809 | paragraph | - | 0.033 | 0.0 | 0.973 |
| 12 | PMC12028424_0 | PMC12028424 | paragraph | - | 0.033 | 0.0 | 0.969 |
| 13 | PMC12082946_24 | PMC12082946 | paragraph | - | 0.032 | 0.0 | 0.958 |
| 14 | PMC11787011_11 | PMC11787011 | paragraph | - | 0.032 | 0.0 | 0.957 |
| 15 | PMC11970461_12 | PMC11970461 | paragraph | - | 0.031 | 0.0 | 0.952 |
| 16 | PMC11997790_21 | PMC11997790 | paragraph | - | 0.029 | 0.0 | 0.939 |
| 17 | PMC11914319_12 | PMC11914319 | paragraph | - | 0.031 | 0.0 | 0.938 |
| 18 | PMC12014972_0 | PMC12014972 | paragraph | - | 0.032 | 0.0 | 0.936 |

## Evidence Text
### #1 - PMC12009809_6 (PMC12009809, paragraph)

*Section:* Treatment for AoC > Surgery

The first-line surgical approach for isolated aortic coarctation is currently the extended end-to-end anastomosis via a left posterolateral thoracotomy, as it avoids the use of patches or allografts and effectively addresses distal aortic arch hypoplasia. Alternative techniques, such as aortoplasty with patch, subclavian flap aortoplasty, and extra-anatomic grafts, were more common in previous decades but are now reserved for specific anatomies. In cases involving aortic arch hypoplasia, median sternotomy should be preferred to facilitate extended aortic arch reconstruction up to the first brachiocephalic vessel.

Complex anatomical cases may require more extensive reconstruction trough a sternotomic access, that currently is needed in approximately 5%–20% of AoC patients (8, 29).

Surgery is typically performed urgently once the diagnosis is confirmed. In premature or very low-birth-weight neonates, weight gain measures may sometimes be considered as palliative strategy (30, 31). However, successful primary surgical repair has been achieved in infants weighing over 1,000 g (32, 33), despite a higher rate of mid-term restenosis. Conversely, delayed diagnosis and/or repair in adulthood is associated with increased mortality (25). In standard settings, mortality and morbidity rates are low, with a 0.54% 30-day mortality (34).

---

### #2 - PMC12028424_11 (PMC12028424, paragraph)


This study shows that the incidence of HT after successful CoA repair was 58.5% and that high serum levels of plasma renin were independently associated with the development of HT on a two-year follow-up.

Although surgical treatment or transcatheter techniques offer excellent short-term results, long-term morbidity and mortality remain higher in this group of patients [1,2]. Systemic hypertension (HT) is commonly reported at follow-up and can persist even if the aortic obstruction is relieved [24,25,26,27]. HT increases the risk of early coronary artery disease, left ventricular hypertrophy, heart failure, and cerebrovascular events [1,2]. Multiple studies report that postoperative HT is an independent risk factor for premature death [5,28,29,30].

---

### #3 - PMC11743609_0 (PMC11743609, paragraph)


Recurrent coarctation of the aorta (re-CoA) is a well-known although not fully understood complication after surgical repair, typically occurring in 10%–20% of cases within months after discharge.

---

### #4 - PMC12082946_17 (PMC12082946, paragraph)


In our series, 16 (64%) cases developed progressive obstruction on follow-up; nevertheless, obstruction was mild/moderate, requiring no intervention at the time of the last surveillance. Mild obstructive lesions were encountered at the initial diagnosis in two cases, with no intervention required, and slowly progressed to the moderate obstructive degree at a single level. Nine cases with initial COA/arch repair during the neonatal period slowly progressed to mild re-coarctation or were associated with mild to moderate mitral/aortic valvular stenosis or mild subaortic stenosis. Two patients had SAM resection, one with coarctation repair and the other as an isolated procedure, which later demonstrated reemergence of the membrane with minimal obstruction. Another case with a mildly stenotic parachute-like mitral valve and bicuspid aortic valve with tiny SAM underwent a transcatheter device occlusion of a residual PDA after surgical closure with only a moderate increase in the valvular stenosis. One patient with pulmonary hypertension resolved after a surgical operation at 15 months for mitral valve repair, SAM resection, and myomectomy had residual mild mitral regurgitation and mild stenosis, which progressed to moderate stenosis and regurgitation on the last follow-up at the age of 30 months. Lastly, another patient with pulmonary hypertension improved after aortic arch reconstruction with VSD and PDA closure in addition to mitral valve repair at five months of age, with only a newly developed mild mitral stenosis on the last surveillance at the age of 13 months. Regarding morbidity, 16% of cases required admission that was unrelated to elective surgery, and 28% of cases had persistent systemic hypertension after surgical correction. Mortality was encountered in 3 (12%), with two before scheduled intervention, one due to uncontrolled heart failure, and the other due to complications of prematurity in addition to heart failure.

---

### #5 - PMC11970461_18 (PMC11970461, paragraph)


Concomitantly to the reintervention of the LAVV in 21/55 procedures, at least one other area was addressed: RAVV re-repair or replacement in 7 cases, residual ASD or VSD closure in 9 cases, RVOTO relief in 2 cases, and LVOTO relief in 6 cases.

Overall, LVOTO relief was performed in 8 (3.2%) patients, of whom 4 required a second relief for recurrence. At the time of the last follow-up, 5 patients showed echocardiographic turbulent flow signs of recurrence of LVOTO not needing reintervention. In 3 (6.25%) patients reoperations related to TOF such as conduit or pulmonary valve replacement were needed.

The need for early and late reoperations were nearly halved over the surgical eras from 9.1 to 4.4% and 21.2 to 12.3%, respectively. Patients undergoing reoperation for LAVVR had significantly reduced long-term survival compared with no reoperation (log-rank p < 0.001 as depicted in Fig. 2B ).

While patients with CoA and ToF had significantly higher reoperation rates ( p = 0.03 and <0.001, respectively) survival analysis of this subgroup was comparable to the entire cohort. Independent risk factors for reoperation were complex AVSD ( p = 0.018) and LAVVR ≥ I–II° after repair ( p = 0.002; Table 3 ).

---

### #6 - PMC11787011_5 (PMC11787011, paragraph)


Coarctation of the aorta (CoA) is a congenital narrowing of the thoracic aorta, often located near the ligamentum arteriosum [1]. Less frequently, the obstruction may be situated in the abdominal aorta or present as diffuse arch hypoplasia with a long, narrowed segment proximal to the left subclavian artery [2]. It is commonly associated with bicuspid aortic valve and diffuse arteriopathy. In most cases, the initial presentation is characterized by upper extremity systolic hypertension [3]. Although it can be diagnosed early, up to 20 % of cases may remain undetected until adulthood [4]. The prognosis for untreated CoA is very poor, with mortality rates as high as 90 % before the age of 50 [5]. The usual treatment involves percutaneous intervention or surgical repair. Both interventions have been demonstrated to greatly improve survival and prognosis when performed in a timely fashion [6,7]. The advancement of balloon and stent designs has led to a notable replacement of surgical repair with percutaneous coarctoplasty [8]. However, complications including recoarctation, aneurysm formation, and stent dislodgment, may occur and necessitate continued monitoring and potential additional interventions [9]. We describe a unique case of aortic coarctation wherein the stent was dislodged from the balloon during the intervention. This case report presented in line with the SCARE criteria [10].

---

### #7 - PMC11697624_3 (PMC11697624, paragraph)


Pseudoaneurysm following coarctation of the aorta (CoA) repair represents a rare but severe complication. The reported incidence ranges from 1.3 to 3.0% [1], and it may occur regardless of the surgical technique utilized [2]. The exact cause of pseudoaneurysm formation remains unclear; however, several contributing factors have been suggested, including infection, hypertension, congenital aortic wall weakness, and high-velocity jet streams through the coarcted segment [3, 4]. Here, we present a case of an adult of pseudoaneurysm with a history of CoA repair 17 years prior, who was initially misdiagnosed as aortic dissection (AD).

---

### #8 - PMC11914319_10 (PMC11914319, paragraph)


Coarctation of the aorta is a common congenital heart defect, with an incidence of 0.3–0.4 per 1000 live births.3–5 It often occurs in isolation but is frequently associated with other congenital anomalies, such as mitral valve abnormalities (60%), aortic arch hypoplasia and other arch defects (18%), ventricular septal defects (13%), and subaortic stenosis (6%).6

Treatment options for aortic coarctation include surgery, balloon angioplasty, and stenting.7 A key observational study by Forbes et al. in 2011, conducted by the Consortium for Congenital Cardiovascular Intervention Research, compared the safety and efficacy of these interventions in patients with congenital aortic stenosis. The study found that stenting outperformed both surgical repair and balloon angioplasty in terms of safety and effectiveness, with the balloon angioplasty group experiencing a significantly higher rate of vessel-related complications.8,9

---

### #9 - PMC11997790_2 (PMC11997790, paragraph)


Thirty consecutive patients were included (2006–24). Median age and weight were 6.0 [interquartile range: 4.0–7.8)] days and 3.1 (2.7–3.5) kg, respectively. Simple congenital heart disease with simple intracardiac shunts (n = 17) and complex congenital heart disease (Complete Atrioventricular Septal Defect (AVSD), interrupted aortic arch and univentricular hearts) (n = 13) constituted the cohort. Non-ischaemic clamp time for roof enlargement was 43 (36–50) min. Ischaemic clamp time for coarctation resection was 23 (21–25) min. Pulmonary artery banding was performed in 19 (63.3%) patients. Twenty-seven (90%) successfully underwent staged repair at 6.1 (4.5–8.2) months age. Follow-up was complete at a median duration of 46.9 (21.7–159.9) months. All patients survived the operation and are in good health at follow-up. Median ventilation time, ICU and hospital stay were 1 (1–2), 3 (2–5) and 23.5 (14–40) days, respectively. No patient developed any neurological complication. Three developed left subclavian artery thrombosis, one requiring surgical revision. With one unrelated late accidental death 14 years after neonatal repair, Kaplan–Meier survival was 90.9 [50.8–98.7]% at 15 years. Two patients underwent arch re-enlargement at the inner curvature to accommodate the DKS during stage 2, resulting in freedom from reoperation of 93.3[75.9–98.3]% at 10 years. All survivors enjoy subjective normal exercise tolerance with no relevant gradient. No patient is on anti-hypertensive medication. Median Z value of the proximal, transverse and distal arch was normalized to −0.88 (−2.19 to −0.12), −0.66(−1.33 to 0.08) and 0.34 (−0.10 to 1.33), respectively, at the last follow-up. Twenty-three (76.7%) arches achieved Romanesque shape at follow-up.

---

### #10 - PMC12014972_3 (PMC12014972, paragraph)


Coarctation of the aorta (CoA) accounts for 5–8% of all congenital heart defects. Transcatheter stent implantation has become the standard of care for native or recurrent coarctation [1]. Stent migration is a potential procedure complication observed in approximately 5% of cases [2]. Although proximal stent migration usually requires surgical intervention [3, 4], transcatheter management has been described in few cases [5, 6]. We describe a case of successful transcatheter management of this unusual complication.

---

### #11 - PMC12009809_7 (PMC12009809, paragraph)


Complications, including left recurrent laryngeal nerve injury, bronchial compression, early re-coarctation, and paradoxical hypertension, occur in approximately 5% of patients. Older age at repair (>20 years) and preoperative hypertension are associated with decreased survival rates (10). Patients younger than 9 years at the time of repair showed significantly lower rates of hypertension at 5–15 years of follow-up. Additionally, younger age at repair and end-to-end anastomosis correction are linked to fewer reintervention on the descending aorta.

---

### #12 - PMC12028424_0 (PMC12028424, paragraph)


Background: Coarctation of the aorta (CoA) represents 5% to 7% of all congenital heart diseases. Surgery and interventional methods offer great short-term results, but the occurrence of postoperative hypertension associated with cardiovascular and cerebral vascular disease increases mortality and morbidity in the long term. This study aims to investigate risk factors associated with postoperative hypertension in pediatric patients with early repair of isolated aortic coarctation. Subjects and Methods: A total of 41 patients with isolated aortic coarctation were included. The mean age was 35.3 ± 46.34 days. Early repair under one month was performed in 65.9% of patients. In all except two patients, end-to-end anastomosis was used. A follow-up at two years revealed an incidence of 58.5% of hypertension. Using logistic regression, preoperative renin plasma concentration above the upper normal level (46.1 μUI/mL) was independently associated with the occurrence of hypertension (OR = 2.49, 95% CI = 2.001–5.03, p = 0.001). Conclusion: Coarctation of the aorta is not just a simple mechanical obstruction of the aorta and should be seen and managed as a systemic disease. Abnormal preoperative renin concentrations were independently associated with the occurrence of HT at follow-up, suggesting that vascular dysfunction could play a role in hypertension development after successful CoA repair, negatively influencing the long-term prognostic of these patients.

---

### #13 - PMC12082946_24 (PMC12082946, paragraph)


In 76% of our patients, surgical interventions were performed, and 16% had transcatheter interventions. The most common surgical procedure was COA repair, performed in 56% of patients with or without aortic arch reconstruction, followed by SAM resection in 16% of cases. The most common percutaneous intervention was balloon aortic valvuloplasty, accounting for 8%. Consistent results were demonstrated by Nicholson et al., who reported surgical interventions in 82% of patients, with 18% requiring catheterization. The most frequent initial surgical interventions in their series were COA repair and subaortic resection [14]. Similarly, Aslam et al. reported that 82% of patients underwent cardiac interventions during childhood. The most common initial interventions were COA repair, relief of LV outflow tract obstruction, and mitral valve repair or replacement [3]. However, Ma et al. reported a significantly lower prevalence of surgery in only 34% of cases, and 8% underwent catheter interventions [18]. Reintervention was performed in 12% of cases in the current series. In contrast, Brown et al. reported a higher reintervention rate in 66% of cases and a 31% reintervention rate in the Aslam et al. series [3, 15].

---

### #14 - PMC11787011_11 (PMC11787011, paragraph)


Coarctation of the aorta (COA) accounts for 6–8 % of all congenital heart disease cases and is one of the most common congenital cardiac pathologies [2,11]. Significant native or recurrent aortic coarctation is defined as a resting peak-to-peak gradient from the upper to the lower extremity of >20 mmHg or a mean Doppler systolic gradient of >20 mmHg, or a gradient of >10 mmHg in combination with either impaired left ventricular systolic function or aortic regurgitation, or with collateral flow [12].

The presence of any of the following is among indications of surgical or transcatheter intervention (including balloon angioplasty and stenting): a peak-to-peak systolic pressure gradient exceeding 20 mmHg at the site of the coarctation; the existence of significant collateral flow; systemic hypertension; and heart failure resulting from the coarctation [13]. Regarding the presented case, the indications for intervention were as follows: symptoms, hypertension, an upper extremity/lower extremity resting peak-to-peak gradient of >20 mmHg, and a mean Doppler systolic gradient of >20 mmHg [[13], [14], [15]].

---

### #15 - PMC11970461_12 (PMC11970461, paragraph)


Mean bypass time was 122 minutes ± 43 and mean cross-clamp time was 71 minutes ± 26. In 16.5% of patients, partial cleft closure was performed due to marked hypoplasia of valvular tissue. Concomitant surgical correction for right ventricular outflow tract obstruction or coarctation was needed in 6.5 and 1.6% of patients, respectively.

---

### #16 - PMC11997790_21 (PMC11997790, paragraph)


Many groups espouse a philosophy of single-stage complete repair using CPB, through a median sternotomy approach. This involves the use of antegrade cerebral perfusion or deep hypothermia and circulatory arrest (DHCA) [21, 22]. Tsang et al. [16] reported 49.7% rate of sternotomies with DHCA as a repair strategy for CoA and severe arch hypoplasia. While, from a purely anatomic-surgical perspective, such a strategy allows a predictable correction of any level of aortic obstruction, those associated with intracardiac repair carry significantly higher risk of morbidity (neurological and constitutional) and mortality. The early mortality of a sternotomy approach has been reported to range from 12.5 to 14.05% [15, 22] with actuarial survival of 75[65–83]% at 12 months and actuarial freedom from recurrent arch obstruction of 69[48–85]% at 46 months.

---

### #17 - PMC11914319_12 (PMC11914319, paragraph)


In our clinical experience with this combined approach—using a NuMed CP® BES alongside a Medtronic® SES—the results have been promising. We recommend the adoption of this treatment approach for patients who meet the following criteria: (i) age > 65 years; (ii) surgical risk > 6%; (iii) complex aortic anatomy, such as coarctation located at the aortic arch, coexistence of aortic coarctation and aortic dilation, or stenosis at the branching points of the aorta; (iv) stenosis that cannot be resolved by single stent implantation; (v) patients at risk of rupture due to fragile arterial walls or extensive calcified atherosclerotic plaques; and (vi) patients with concurrent aneurysms or aortic dissections. For cases where the stent obstructs blood flow to arterial branches, solutions such as the creation of a bypass or stent fenestration can be employed. However, this approach has only been tested in a single patient, and further experience and long-term follow-up are necessary to fully assess its efficacy and safety.

---

### #18 - PMC12014972_0 (PMC12014972, paragraph)


Transcatheter stenting has become the preferred treatment for native and recurrent coarctation of aorta (CoA), but complications such as stent migration occur in approximately 5% of cases. Proximal stent migration is particularly challenging and often requires surgical intervention. This report highlights the successful transcatheter management of proximal stent migration during CoA stenting in a high-risk patient.

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
        "PMC12009809",
        "PMC12014972",
        "PMC11970464",
        "PMC11787011",
        "PMC12028424",
        "PMC11697624",
        "PMC11914319",
        "PMC12055418",
        "PMC11879088"
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
        "PMC11970464",
        "PMC11854409",
        "PMC12028424",
        "PMC11810876",
        "PMC11836792",
        "PMC12025643",
        "PMC11997790",
        "PMC11970461",
        "PMC11729957"
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
        "PMC12009809",
        "PMC12014972",
        "PMC12082946",
        "PMC12028424",
        "PMC11697624",
        "PMC11787011",
        "PMC11970464",
        "PMC12055418",
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
        "PMC11997790",
        "PMC11743609",
        "PMC12009809",
        "PMC12034379",
        "PMC11869840",
        "PMC11828486",
        "PMC12086954",
        "PMC11970464",
        "PMC11747392",
        "PMC11810876"
      ]
    },
    "q5": {
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
        "PMC11970464",
        "PMC11787011",
        "PMC12028424",
        "PMC11697624",
        "PMC11914319",
        "PMC12055418",
        "PMC11879088"
      ]
    },
    "q6": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11743609",
        "PMC11970464",
        "PMC11854409",
        "PMC12028424",
        "PMC11810876",
        "PMC11836792",
        "PMC12025643",
        "PMC11997790",
        "PMC11970461",
        "PMC11729957"
      ]
    },
    "q7": {
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
        "PMC12082946",
        "PMC12028424",
        "PMC11697624",
        "PMC11787011",
        "PMC11970464",
        "PMC12055418",
        "PMC11914319"
      ]
    },
    "q8": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11997790",
        "PMC11743609",
        "PMC12009809",
        "PMC12034379",
        "PMC11869840",
        "PMC11828486",
        "PMC12086954",
        "PMC11970464",
        "PMC11747392",
        "PMC11810876"
      ]
    },
    "q9": {
      "known_papers": [],
      "recall@5": 0.0,
      "recall@10": 0.0,
      "recall@20": 0.0,
      "mrr": 0.0,
      "found_in_ranking": [],
      "paper_ranking_top10": [
        "PMC11970461",
        "PMC11743609",
        "PMC11810876",
        "PMC11886819",
        "PMC12009809",
        "PMC12028424",
        "PMC11776063",
        "PMC11787011",
        "PMC11729957",
        "PMC11927473"
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
    },
    "H2": {
      "covered": true
    },
    "H3": {
      "covered": true
    },
    "H4": {
      "covered": true
    }
  },
  "final_evidence_chunk_recall": {},
  "n_final_evidence": 18,
  "note": "Paper recall and MRR are computed against branch-level known papers (spec 40-41); requirement recall is the fraction of hops with evidence in the final set."
}
```