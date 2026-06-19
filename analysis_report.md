# Model / Strategy Comparison & Analysis Report
**Project: Email Generation Assistant**

This report compares two email generation configurations using 10 diverse scenarios.

---

## 1. Quantitative Performance Summary

The 10 scenarios in `scenarios.json` were evaluated against the three custom metrics. Below are the final average results:

| Metric | Config A: Few-Shot | Config B: Chain-of-Thought | Difference (B - A) |
|---|---|---|---|
| **Fact Recall** | 100.00% | 100.00% | 0.00% |
| **Tone Accuracy** | 100.00/100 | 95.00/100 | -5.00 |
| **Conciseness & Fluency** | 100.00/100 | 100.00/100 | 0.00 |
| **Overall Average** | 100.00/100 | 98.33/100 | -1.67 |

*Evaluation Mode: Live API Evaluation*
*Model Used: openai/gpt-oss-120b:free*

---

## 2. Model / Strategy Comparison & Analysis

### 1. Which model/strategy performed better according to the 3 custom metrics?
Based on the metrics above, **Config A (Few-Shot) (100.00/100)** performed better overall than **Config B (Chain-of-Thought) (98.33/100)**, a margin of 1.67 points.
- **Fact Recall**: Both configurations scored equally (100.00%), so this metric did not differentiate them.
- **Tone Accuracy**: Config A (Few-Shot) scored 100.00/100, 5.00 points higher than Config B (Chain-of-Thought) (95.00/100).
- **Conciseness & Fluency**: Both configurations scored equally (100.00/100), so this metric did not differentiate them.

### 2. What was the biggest failure mode of the lower-performing config, based on the data?
The biggest weakness of the lower-performing config (**Config B (Chain-of-Thought)**) was on **Tone Accuracy**, where it trailed Config A (Few-Shot) by 5.00 points (95.00 vs 100.00). This was the metric that most dragged down its overall score.

### 3. Which would you recommend for production and why, justified with the metric data?
Based on this data, we recommend **Config A (Few-Shot)** for production. It achieved the higher overall score (100.00/100 vs 98.33/100, a 1.67-point lead) and was at least as strong on every individual metric that mattered. It also has the practical advantage of lower token usage and latency than Chain-of-Thought, making it cheaper to run at scale.

## 3. Prompt Templates Used

### Config A: Few-Shot Prompting Templates

#### System Prompt:
```text
You are a professional email assistant.
Your task is to generate a complete email (including a subject line, greeting, body, and sign-off) based on the provided intent, key facts, and desired tone.

You must:
1. Include every key fact naturally. Do not leave any fact out.
2. Match the requested tone exactly.
3. Output a complete email with a subject line, greeting, body, and sign-off.
4. Return ONLY the email text. Do not include any conversational preamble, introduction, markdown code block fences (like ```), or comments.
```

#### User Template:
```text
### Examples

Example 1:
- Intent: Ask for an update on a graphic design contract.
- Key Facts:
  * Design agency is "PixelCraft"
  * Contract was sent on May 12th
  * Project start date is June 1st
- Tone: professional
- Output Email:
Subject: Inquiry regarding PixelCraft graphic design contract

Dear PixelCraft Team,

I hope this email finds you well. I am writing to request an update on the graphic design contract that we sent over on May 12th.

As our proposed project start date of June 1st is approaching, we would like to finalize the paperwork as soon as possible. Please let us know if you have any questions or require any adjustments to the agreement.

We look forward to working together.

Best regards,
Aria Sterling
Operations Director

Example 2:
- Intent: Invite a colleague to a casual lunch.
- Key Facts:
  * Lunch is scheduled for Friday at 12:30 PM
  * Location is "The Green Bistro"
  * Celebrating Mark's promotion
- Tone: casual
- Output Email:
Subject: Lunch this Friday? (Celebrating Mark!)

Hey team,

Hope you're having a great week!

Just a quick heads-up that we're planning a casual lunch this Friday at 12:30 PM at The Green Bistro. We'll be celebrating Mark's recent promotion, so it should be a fun time.

Let me know if you can make it so I can lock in the reservation. Hope to see you there!

Best,
Sam

### Your Task
Generate an email for:
- Intent: {intent}
- Key Facts:
{key_facts_bullets}
- Tone: {tone}
- Output Email:
```

### Config B: Chain-of-Thought Prompting Templates

#### System Prompt:
```text
You are a professional email assistant.
Your task is to generate a complete email based on the provided intent, key facts, and desired tone.

You must follow these steps in your reasoning:
1. Identify the target audience and what relationship tone is appropriate.
2. Outline the structure of the email (Subject, Greeting, Body, Sign-off).
3. Plan how to naturally integrate every single key fact from the list.
4. Verify that the planned email matches the requested tone.

Format your output exactly as follows:
<reasoning>
Provide your step-by-step reasoning here, detailing:
- Audience & Tone analysis
- Structure plan
- Key facts integration checklist
</reasoning>

<email>
Subject: [Your Subject Line]

[Greeting]

[Body of the email including all key facts naturally integrated]

[Sign-off]
</email>

Do not include any other text, markdown code blocks, or preamble outside of these XML tags.
```

#### User Template:
```text
Generate an email for:
- Intent: {intent}
- Key Facts:
{key_facts_bullets}
- Tone: {tone}
```

## 4. Definitions and Logic for the 3 Custom Metrics

### Fact Recall
```text
FACT RECALL (Automated LLM-as-a-judge):
Measures the fraction of the scenario's key facts present in the generated email.
An LLM judge reviews the email and, for each key fact, decides whether it is clearly
conveyed (yes/no). The final score is computed as (facts present / total facts) * 100.
```

### Tone Accuracy
```text
TONE ACCURACY (LLM-as-a-judge):
Evaluates how well the generated email matches the requested tone. An LLM judge rates the
tone on a scale of 1 to 5. The judge outputs a JSON response containing 'score' and 'reason'.
The raw score is normalized to a 0-100 scale using the formula: (score - 1) / 4 * 100.
```

### Conciseness Fluency
```text
CONCISENESS & FLUENCY (Hybrid Python + LLM):
Combines (a) a readability/length component computed locally in Python with (b) an LLM judge
grammar and fluency rating (1-5, normalized to 0-100). The two scores are averaged 50/50.
Python Readability/Length Formula:
- Word Count (W) Score: 100 if 80 <= W <= 220. If W < 80, penalty is (80 - W) * 2. If W > 220, penalty is (W - 220) * 0.8.
- Sentence Length (S) Score: 100 if 10 <= S <= 20. If S < 10, penalty is (10 - S) * 10. If S > 20, penalty is (S - 20) * 5.
- Python Score = (Word Count Score + Sentence Length Score) / 2
LLM Fluency Formula:
- Normalized Fluency Score = (LLM Score - 1) / 4 * 100
Final Score = (Python Score + Normalized Fluency Score) / 2
```


## 5. Raw Scenario Evaluation Data

### Config A: Few-Shot Strategy Raw Scores

| Scenario ID | Target Tone | Fact Recall (%) | Tone Accuracy (/100) | Conciseness/Fluency (/100) | Overall (/100) |
|---|---|---|---|---|---|
| scenario_1 | professional | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_2 | confident | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_3 | firm | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_4 | sincere | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_5 | enthusiastic | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_6 | formal | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_7 | warm | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_8 | urgent | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_9 | appreciative | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_10 | assertive | 100.00 | 100.00 | 100.00 | 100.00 |

### Config B: Chain-of-Thought Strategy Raw Scores

| Scenario ID | Target Tone | Fact Recall (%) | Tone Accuracy (/100) | Conciseness/Fluency (/100) | Overall (/100) |
|---|---|---|---|---|---|
| scenario_1 | professional | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_2 | confident | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_3 | firm | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_4 | sincere | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_5 | enthusiastic | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_6 | formal | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_7 | warm | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_8 | urgent | 100.00 | 50.00 | 100.00 | 83.33 |
| scenario_9 | appreciative | 100.00 | 100.00 | 100.00 | 100.00 |
| scenario_10 | assertive | 100.00 | 100.00 | 100.00 | 100.00 |
