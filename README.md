# Data-Driven Decisions

A practical guide to **making better decisions with data** — covering when to use data (and when not to), statistical literacy for non-statisticians, common pitfalls, and templates for structuring data-informed analysis.

## Table of Contents

- [Data-Driven vs. Data-Informed](#data-driven-vs-data-informed)
- [The Decision Data Lifecycle](#the-decision-data-lifecycle)
- [Statistical Concepts for Decision-Makers](#statistical-concepts-for-decision-makers)
- [Frameworks](#frameworks)
  - [1. Hypothesis-Driven Analysis](#1-hypothesis-driven-analysis)
  - [2. A/B Testing Decision Framework](#2-ab-testing-decision-framework)
  - [3. Base Rate Analysis](#3-base-rate-analysis)
  - [4. Expected Value Calculation](#4-expected-value-calculation)
  - [5. Fermi Estimation](#5-fermi-estimation)
  - [6. Pre-Registration Template](#6-pre-registration-template)
- [Data Pitfalls](#data-pitfalls)
- [When NOT to Use Data](#when-not-to-use-data)
- [Templates](#templates)
- [Resources](#resources)
- [License](#license)

---

## Data-Driven vs. Data-Informed

| Approach | Description | When to Use |
|----------|-------------|------------|
| **Data-driven** | Let data determine the decision | Repeatable, measurable, low-stakes decisions |
| **Data-informed** | Use data as ONE input alongside judgment, context, values | Complex, strategic, value-laden decisions |

**Key insight:** Most important decisions should be data-*informed*, not data-*driven*. Data tells you what happened, not why it happened or what you should value.

---

## The Decision Data Lifecycle

```
1. QUESTION — What decision are we making?
      ↓
2. HYPOTHESIS — What do we expect and why?
      ↓
3. DATA — What data do we need? Do we have it? Is it reliable?
      ↓
4. ANALYSIS — What does the data actually say?
      ↓
5. INTERPRETATION — What does it mean in context?
      ↓
6. DECISION — What do we do based on this + other inputs?
      ↓
7. REVIEW — Did the decision produce the expected outcome?
```

**Most teams skip steps 1, 2, and 7.** They jump to data collection without clear questions, analyze without hypotheses (data fishing), and never check if their data-based decisions actually worked.

---

## Statistical Concepts for Decision-Makers

You don't need a statistics degree. You need these concepts:

### Correlation vs. Causation

**Correlation:** X and Y move together.
**Causation:** X causes Y.

Ice cream sales and drowning deaths are correlated (both rise in summer). Ice cream doesn't cause drowning. The lurking variable is temperature.

**To establish causation, you need:**
- Randomized controlled experiment (gold standard)
- Or careful causal inference methods (instrumental variables, regression discontinuity, difference-in-differences)

### Statistical Significance vs. Practical Significance

- **Statistical significance:** "This result is unlikely to be due to chance" (p < 0.05)
- **Practical significance:** "This result is large enough to matter"

A website change that increases conversion by 0.001% might be statistically significant with enough traffic, but practically meaningless. Always ask: "Is the effect SIZE meaningful?"

### Base Rates

The general frequency of an event. Before evaluating specific evidence, know the base rate.

**Example:** A test is 95% accurate. You test positive for a disease that affects 1 in 10,000 people. What's the probability you have it?

Not 95%. It's about 0.2%. The base rate (1/10,000) matters enormously.

### Regression to the Mean

Extreme performance (good or bad) tends to be followed by more average performance. This isn't because of any action taken — it's mathematical reality.

**Implication:** Don't attribute improvement after a bad quarter to your intervention. It might just be regression to the mean.

### Survivorship Bias

You see the winners. You don't see the losers who did the same thing.

**Example:** "Successful CEOs dropped out of college, so dropping out leads to success." You don't see the millions who dropped out and struggled.

### Simpson's Paradox

A trend that appears in subgroups reverses when the groups are combined.

**Classic example:** Treatment A appears better for both men and women separately, but Treatment B appears better overall — because the groups are different sizes with different baseline rates.

**Takeaway:** Always look at data at multiple levels of aggregation.

---

## Frameworks

### 1. Hypothesis-Driven Analysis

**Don't:** "Let's look at the data and see what we find"
**Do:** "We believe X because of Y. Let's test that belief."

```markdown
## Hypothesis-Driven Analysis: [Question]

### Hypothesis
"We believe [specific claim] because [reasoning]."

### Key assumptions
1.
2.
3.

### What data would support this hypothesis?

### What data would disprove this hypothesis?

### Data collected:

### Conclusion:
- [ ] Hypothesis supported — evidence: ____
- [ ] Hypothesis rejected — evidence: ____
- [ ] Inconclusive — need more data on: ____
```

---

### 2. A/B Testing Decision Framework

```markdown
## A/B Test Plan: [Feature/Change]

### What are we testing?
Change: ____
Control: ____

### Hypothesis
"Changing [X] will improve [metric] by [amount] because [reason]."

### Primary metric: ____
### Guardrail metrics (must not degrade): ____

### Sample size needed: ____
### Test duration: ____
### Statistical significance threshold: p < 0.05

### Decision criteria
- If lift > [X]% with significance → Ship it
- If lift < [X]% → Don't ship
- If inconclusive → [extend / re-test / move on]

### Results
| Metric | Control | Variant | Lift | Significant? |
|--------|---------|---------|------|:-:|
| | | | | |

### Decision: ____
```

---

### 3. Base Rate Analysis

Before making a specific prediction, anchor to the base rate.

```markdown
## Base Rate Check: [Decision/Prediction]

### My specific prediction: ____

### Base rate: What happens in similar situations?
- General success rate for this type of [project/investment/hire]: ____%
- Source of base rate: ____

### What makes my situation different?
- Factors above base rate: ____
- Factors below base rate: ____

### Adjusted estimate: ____%

### Is this adjustment justified or am I just being overconfident?
```

---

### 4. Expected Value Calculation

```markdown
## Expected Value: [Decision]

### Options and outcomes:

#### Option A: ____
| Outcome | Probability | Value ($) | Expected Value |
|---------|:-:|:-:|:-:|
| Best case | __% | $ | $ |
| Base case | __% | $ | $ |
| Worst case | __% | $ | $ |
| **Total EV** | 100% | | **$** |

#### Option B: ____
| Outcome | Probability | Value ($) | Expected Value |
|---------|:-:|:-:|:-:|
| Best case | __% | $ | $ |
| Base case | __% | $ | $ |
| Worst case | __% | $ | $ |
| **Total EV** | 100% | | **$** |

### Decision: Choose option with highest EV, adjusted for risk tolerance.
### Risk check: Can I survive the worst case?
```

---

### 5. Fermi Estimation

Estimate unknown quantities by breaking them into smaller, estimable components.

**Example: "How many piano tuners are in Chicago?"**
```
Chicago population: ~2.7 million
People per household: ~2.5
Households: ~1.1 million
% with pianos: ~5%
Pianos: ~55,000
Tunings per piano per year: ~1
Tunings per tuner per day: ~4
Working days per year: ~250
Tunings per tuner per year: ~1,000
Piano tuners needed: 55,000 / 1,000 = ~55

Actual answer: ~100 (within an order of magnitude — good Fermi estimate)
```

**Template:**
```markdown
## Fermi Estimate: [Question]

### Decomposition
| Component | Estimate | Reasoning |
|-----------|---------|-----------|
| | | |
| | | |
| | | |

### Calculation
[Show the math]

### Result: ____
### Confidence range: [lower bound] to [upper bound]
```

---

### 6. Pre-Registration Template

Commit to your analysis plan BEFORE looking at the data. Prevents p-hacking and motivated reasoning.

```markdown
## Pre-Registration: [Analysis]

### Research question: ____
### Hypothesis: ____
### Primary outcome variable: ____
### Analysis method: ____
### Sample size / data source: ____
### Exclusion criteria (decided in advance): ____
### What would "support" look like?: ____
### What would "reject" look like?: ____

Date registered: ____
Analyst: ____
```

---

## Data Pitfalls

| Pitfall | Description | Prevention |
|---------|-------------|-----------|
| **Cherry-picking** | Selecting data that supports your conclusion | Pre-register hypotheses |
| **p-hacking** | Running many tests until something is "significant" | Correct for multiple comparisons |
| **Confusing correlation and causation** | Assuming X causes Y because they correlate | Run experiments; consider confounders |
| **Survivorship bias** | Analyzing only successes | Include failures in your dataset |
| **Goodhart's Law** | "When a measure becomes a target, it ceases to be a good measure" | Use multiple metrics; monitor gaming |
| **Overfitting** | Model fits historical data perfectly but predicts poorly | Out-of-sample testing; simpler models |
| **Selection bias** | Sample isn't representative of the population | Randomize; check sample composition |
| **Anchoring to dashboards** | Treating dashboards as reality | Talk to customers; visit the front lines |
| **McNamara Fallacy** | Measuring only what's easily measurable | Acknowledge what's important but unmeasured |

---

## When NOT to Use Data

Data is powerful but not omniscient. Don't use data for:

| Situation | Why Data Falls Short | Better Approach |
|-----------|---------------------|----------------|
| **Unprecedented decisions** | No historical data exists | First principles, scenario planning |
| **Value judgments** | Data says "what is," not "what should be" | Ethical frameworks, values discussion |
| **Small-sample decisions** | Statistical noise dominates signal | Expert judgment, qualitative analysis |
| **Breakthrough innovation** | Customers can't tell you what doesn't exist yet | Vision, design thinking |
| **Emotional/relationship decisions** | Important dimensions aren't quantifiable | Wisdom, empathy, reflection |

---

## Templates

### Data-Informed Decision Document

```markdown
# Decision: [Topic]

## The Question
What specific decision are we making?

## What the Data Says
| Metric / Finding | Value | Source | Confidence |
|-----------------|-------|--------|:-:|
| | | | H/M/L |
| | | | H/M/L |

## What the Data Doesn't Tell Us
- Unmeasured factors:
- Data limitations:
- Possible confounders:

## Other Inputs
- Expert opinion:
- Customer qualitative feedback:
- Strategic context:
- Values and ethics:

## Recommendation
Based on data + context + judgment:

## How We'll Measure Success
- Primary metric:
- Timeline:
- Review date:
```

---

## Resources

**Books:**
- *Thinking, Fast and Slow* — Daniel Kahneman
- *How to Lie with Statistics* — Darrell Huff
- *Naked Statistics* — Charles Wheelan
- *Superforecasting* — Philip Tetlock
- *The Signal and the Noise* — Nate Silver

**For real-world decision scenarios that combine data analysis with proven mental models, visit [KeepRule Scenarios](https://keeprule.com/en/scenarios) — interactive exercises for sharpening your judgment.**

---

## Contributing

Have a framework or cautionary tale to add? PRs welcome.

## License

MIT License — see [LICENSE](LICENSE) for details.
