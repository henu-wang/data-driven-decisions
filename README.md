# Data-Driven Decisions

> A practical guide to making better decisions using statistics, A/B testing, Bayesian reasoning, and Python — without a PhD in statistics.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Most decision-making resources focus on frameworks and heuristics. This guide bridges the gap between intuition and evidence by teaching you to use data effectively. Includes working Python code examples you can adapt to your own analyses.

---

## Table of Contents

- [Why Data-Driven Decision Making](#why-data-driven-decision-making)
- [Part 1: Statistical Foundations](#part-1-statistical-foundations)
  - [Descriptive Statistics That Actually Matter](#descriptive-statistics-that-actually-matter)
  - [Understanding Distributions](#understanding-distributions)
  - [Correlation vs Causation](#correlation-vs-causation)
  - [Statistical Significance Explained Simply](#statistical-significance-explained-simply)
- [Part 2: A/B Testing](#part-2-ab-testing)
  - [When to A/B Test](#when-to-ab-test)
  - [Designing a Valid Experiment](#designing-a-valid-experiment)
  - [Sample Size Calculator](#sample-size-calculator)
  - [Analyzing Results](#analyzing-results)
  - [Common A/B Testing Mistakes](#common-ab-testing-mistakes)
- [Part 3: Bayesian Thinking](#part-3-bayesian-thinking)
  - [Bayes Theorem in Plain English](#bayes-theorem-in-plain-english)
  - [Updating Beliefs with Evidence](#updating-beliefs-with-evidence)
  - [Bayesian A/B Testing](#bayesian-ab-testing)
- [Part 4: Decision Analysis](#part-4-decision-analysis)
  - [Expected Value Calculations](#expected-value-calculations)
  - [Decision Trees](#decision-trees)
  - [Monte Carlo Simulation](#monte-carlo-simulation)
  - [Sensitivity Analysis](#sensitivity-analysis)
- [Part 5: Practical Python Examples](#part-5-practical-python-examples)
  - [Quick Data Exploration Template](#quick-data-exploration-template)
  - [A/B Test Analysis Pipeline](#ab-test-analysis-pipeline)
- [Part 6: Cognitive Biases in Data Interpretation](#part-6-cognitive-biases-in-data-interpretation)
- [Part 7: Building a Data-Driven Culture](#part-7-building-a-data-driven-culture)
- [Cheat Sheets](#cheat-sheets)
- [Further Resources](#further-resources)

---

## Why Data-Driven Decision Making

Not every decision needs data. Use data when:

| Situation | Data Helps | Intuition Suffices |
|-----------|-----------|-------------------|
| High stakes, reversible | Test before committing | |
| High stakes, irreversible | Gather all available evidence | |
| Low stakes, reversible | | Just decide and learn |
| Repeated decisions at scale | Small improvements compound | |
| Novel, one-time decisions | As input, not sole driver | Combined with experience |

The goal is not to eliminate judgment. It is to inform judgment with evidence, reducing the influence of biases and noise.

---

## Part 1: Statistical Foundations

### Descriptive Statistics That Actually Matter

Beyond mean and standard deviation, these are the statistics that drive decisions:

| Statistic | What It Tells You | When to Use It |
|-----------|-------------------|----------------|
| **Median** | Typical value (robust to outliers) | Salaries, response times, revenue per customer |
| **Percentiles (p95, p99)** | Tail behavior | SLA compliance, load testing, risk analysis |
| **Coefficient of variation** | Relative variability (std/mean) | Comparing variability across different scales |
| **Interquartile range** | Spread of the middle 50% | Understanding the "normal" range |

```python
import numpy as np
import pandas as pd

def decision_stats(data: pd.Series) -> dict:
    """Compute the statistics that matter for decisions."""
    return {
        "count": len(data),
        "mean": data.mean(),
        "median": data.median(),
        "std": data.std(),
        "cv": data.std() / data.mean() if data.mean() != 0 else None,
        "p25": data.quantile(0.25),
        "p75": data.quantile(0.75),
        "p95": data.quantile(0.95),
        "p99": data.quantile(0.99),
        "iqr": data.quantile(0.75) - data.quantile(0.25),
    }

# Example: Analyze response times
response_times = pd.Series([120, 135, 128, 142, 118, 155, 890, 132, 127, 141])
stats = decision_stats(response_times)
print(f"Mean: {stats['mean']:.0f}ms, Median: {stats['median']:.0f}ms, P99: {stats['p99']:.0f}ms")
# Mean is inflated by the 890ms outlier; median tells the real story
```

### Understanding Distributions

| Distribution | Shape | Real-World Examples |
|-------------|-------|-------------------|
| Normal | Bell curve | Heights, test scores, manufacturing tolerances |
| Log-normal | Right-skewed | Income, file sizes, time-to-resolution |
| Power law | Extreme right tail | Website traffic, city sizes, wealth |
| Bimodal | Two peaks | Mixed populations (e.g., casual vs power users) |

**Decision implication:** If your data is not normally distributed (most business data is not), the mean is misleading. Always check the distribution before choosing summary statistics.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

def check_distribution(data, name="data"):
    """Quick visual check of data distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'{name} - Histogram')

    # Box plot
    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f'{name} - Box Plot')

    # QQ plot
    stats.probplot(data, dist="norm", plot=axes[2])
    axes[2].set_title(f'{name} - QQ Plot (vs Normal)')

    plt.tight_layout()
    plt.savefig(f'{name}_distribution.png', dpi=100)
    plt.show()
```

### Correlation vs Causation

Three requirements for causal claims:

1. **Correlation**: X and Y are statistically associated
2. **Temporal precedence**: X occurs before Y
3. **No confounders**: No third variable Z explains both X and Y

| Method | Can Establish Causation? | When to Use |
|--------|------------------------|-------------|
| Observational correlation | No | Hypothesis generation |
| Controlled A/B test | Yes (if properly designed) | Product changes, marketing |
| Natural experiment | Sometimes | Policy changes, external shocks |
| Instrumental variables | Sometimes | When randomization is impossible |
| Difference-in-differences | Sometimes | Before/after comparisons with control groups |

### Statistical Significance Explained Simply

**p-value**: The probability of seeing results this extreme if the null hypothesis (no real effect) were true.

- p < 0.05 does NOT mean "there is a 95% chance the effect is real"
- p < 0.05 means "if there were no real effect, we would see results this extreme less than 5% of the time"

**Practical significance vs statistical significance:**

| Scenario | Statistically Significant? | Practically Significant? | Decision |
|----------|--------------------------|-------------------------|----------|
| +0.1% conversion, p=0.01, n=1M | Yes | No | Ignore |
| +5% conversion, p=0.08, n=500 | No | Possibly | Collect more data |
| +3% conversion, p=0.02, n=10K | Yes | Yes | Ship it |

---

## Part 2: A/B Testing

### When to A/B Test

A/B test when:
- The change is measurable with a clear metric
- You have enough traffic to reach significance in a reasonable time
- The decision is reversible
- The cost of being wrong is higher than the cost of testing

Do NOT A/B test when:
- Sample size is too small (fewer than 1,000 per variant for most tests)
- The change is obviously better or worse
- The change affects a tiny fraction of users
- You are testing more than 3 variants (use multi-armed bandits instead)

### Designing a Valid Experiment

```
1. State the hypothesis: "Changing X will improve Y by Z%"
2. Define the primary metric (ONE metric, not five)
3. Define guardrail metrics (metrics that should NOT get worse)
4. Calculate required sample size
5. Determine test duration (minimum 1 full business cycle, usually 1-2 weeks)
6. Set up random, equal-sized groups
7. Run test for the pre-determined duration (DO NOT peek and stop early)
8. Analyze results
```

### Sample Size Calculator

```python
from scipy.stats import norm
import math

def required_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Calculate required sample size per variant for a two-proportion z-test.

    Args:
        baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
        minimum_detectable_effect: Relative change to detect (e.g., 0.10 for 10% lift)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        Required sample size per variant
    """
    p1 = baseline_rate
    p2 = baseline_rate * (1 + minimum_detectable_effect)
    p_avg = (p1 + p2) / 2

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    n = ((z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) +
          z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (p2 - p1) ** 2

    return math.ceil(n)

# Example: 5% baseline conversion, want to detect 10% relative lift
n = required_sample_size(baseline_rate=0.05, minimum_detectable_effect=0.10)
print(f"Required sample size per variant: {n:,}")
# Output: approximately 31,234 per variant
```

### Analyzing Results

```python
from scipy.stats import chi2_contingency, norm
import numpy as np

def analyze_ab_test(
    control_visitors: int,
    control_conversions: int,
    treatment_visitors: int,
    treatment_conversions: int,
    alpha: float = 0.05,
) -> dict:
    """Analyze an A/B test with conversion data."""
    p_control = control_conversions / control_visitors
    p_treatment = treatment_conversions / treatment_visitors
    relative_lift = (p_treatment - p_control) / p_control

    # Chi-squared test
    table = np.array([
        [control_conversions, control_visitors - control_conversions],
        [treatment_conversions, treatment_visitors - treatment_conversions]
    ])
    chi2, p_value, dof, expected = chi2_contingency(table)

    # Confidence interval for the difference
    se = np.sqrt(p_control * (1 - p_control) / control_visitors +
                 p_treatment * (1 - p_treatment) / treatment_visitors)
    z = norm.ppf(1 - alpha / 2)
    ci_lower = (p_treatment - p_control) - z * se
    ci_upper = (p_treatment - p_control) + z * se

    return {
        "control_rate": f"{p_control:.4f}",
        "treatment_rate": f"{p_treatment:.4f}",
        "relative_lift": f"{relative_lift:.2%}",
        "p_value": f"{p_value:.4f}",
        "significant": p_value < alpha,
        "ci_95": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
    }

# Example
result = analyze_ab_test(
    control_visitors=15000, control_conversions=750,
    treatment_visitors=15000, treatment_conversions=825,
)
for k, v in result.items():
    print(f"  {k}: {v}")
```

### Common A/B Testing Mistakes

| Mistake | Why It Is Wrong | Fix |
|---------|----------------|-----|
| Peeking at results daily and stopping early | Inflates false positive rate to 25-30% | Pre-commit to a sample size; use sequential testing if you must peek |
| Testing too many variants | Requires Bonferroni correction; most tests underpowered | Test 2-3 variants maximum |
| Ignoring novelty effects | New UI gets clicks from curiosity, not value | Run tests for at least 2 weeks |
| Segmenting after the fact | You will always find a "significant" segment by chance | Pre-register segments of interest |
| Using the wrong metric | Optimizing clicks when you care about revenue | Define primary metric before starting |

---

## Part 3: Bayesian Thinking

### Bayes Theorem in Plain English

```
P(Hypothesis | Evidence) = P(Evidence | Hypothesis) x P(Hypothesis) / P(Evidence)
```

In plain English:

```
Updated Belief = How likely the evidence is if the hypothesis is true
                 x How likely the hypothesis was before the evidence
                 / How likely the evidence is overall
```

### Updating Beliefs with Evidence

**Example: Should we launch this feature?**

Prior belief: 60% chance the feature will improve retention (based on user research).

We run a small pilot with 200 users:
- Treatment group (100 users): 45% retention
- Control group (100 users): 40% retention

```python
import numpy as np

def bayesian_update_conversion(
    prior_alpha: float, prior_beta: float,
    successes: int, trials: int
) -> tuple:
    """
    Update a Beta prior with observed conversion data.
    Returns posterior (alpha, beta) parameters.
    """
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (trials - successes)
    return posterior_alpha, posterior_beta

# Prior: Beta(6, 4) represents our 60% prior belief
# Treatment: 45 retained out of 100
post_alpha, post_beta = bayesian_update_conversion(6, 4, 45, 100)
posterior_mean = post_alpha / (post_alpha + post_beta)
print(f"Posterior mean retention: {posterior_mean:.2%}")
# Posterior: ~46.4%, pulled toward the data from our 60% prior
```

### Bayesian A/B Testing

Unlike frequentist testing, Bayesian A/B testing directly answers: "What is the probability that B is better than A?"

```python
import numpy as np

def bayesian_ab_test(
    control_successes: int, control_trials: int,
    treatment_successes: int, treatment_trials: int,
    n_simulations: int = 100_000,
) -> dict:
    """
    Bayesian A/B test using Beta-Binomial model.
    Returns probability that treatment is better than control.
    """
    # Using uniform prior Beta(1,1)
    control_samples = np.random.beta(
        control_successes + 1,
        control_trials - control_successes + 1,
        n_simulations
    )
    treatment_samples = np.random.beta(
        treatment_successes + 1,
        treatment_trials - treatment_successes + 1,
        n_simulations
    )

    prob_treatment_better = (treatment_samples > control_samples).mean()
    expected_lift = ((treatment_samples - control_samples) / control_samples).mean()

    return {
        "prob_treatment_better": f"{prob_treatment_better:.2%}",
        "expected_relative_lift": f"{expected_lift:.2%}",
        "risk_of_choosing_treatment": f"{1 - prob_treatment_better:.2%}",
    }

result = bayesian_ab_test(
    control_successes=750, control_trials=15000,
    treatment_successes=825, treatment_trials=15000,
)
for k, v in result.items():
    print(f"  {k}: {v}")
```

---

## Part 4: Decision Analysis

### Expected Value Calculations

```python
def expected_value(outcomes: list) -> float:
    """
    Calculate expected value from (probability, value) pairs.
    Probabilities should sum to 1.0.
    """
    total_prob = sum(p for p, v in outcomes)
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob}, not 1.0"
    return sum(p * v for p, v in outcomes)

# Example: Should we invest $100K in a new feature?
# 30% chance: +$500K revenue
# 50% chance: +$150K revenue
# 20% chance: -$50K (feature flops, some users churned)
ev = expected_value([
    (0.30, 500_000),
    (0.50, 150_000),
    (0.20, -50_000),
])
roi = ev - 100_000
print(f"Expected value: ${ev:,.0f}")
print(f"Expected ROI after $100K investment: ${roi:,.0f}")
```

### Monte Carlo Simulation

When outcomes are uncertain and interrelated, simulate:

```python
import numpy as np

def revenue_simulation(n_simulations: int = 10_000) -> dict:
    """
    Monte Carlo simulation for annual revenue projection.
    Useful when multiple uncertain variables interact.
    """
    np.random.seed(42)

    # Uncertain inputs (each modeled as a distribution)
    monthly_visitors = np.random.normal(50_000, 10_000, n_simulations)
    conversion_rate = np.random.beta(5, 95, n_simulations)  # ~5% mean
    avg_order_value = np.random.lognormal(np.log(50), 0.3, n_simulations)
    months = 12

    # Calculate annual revenue for each simulation
    annual_revenue = monthly_visitors * conversion_rate * avg_order_value * months

    return {
        "mean_revenue": f"${np.mean(annual_revenue):,.0f}",
        "median_revenue": f"${np.median(annual_revenue):,.0f}",
        "p10_revenue": f"${np.percentile(annual_revenue, 10):,.0f}",
        "p90_revenue": f"${np.percentile(annual_revenue, 90):,.0f}",
        "prob_above_1M": f"{(annual_revenue > 1_000_000).mean():.1%}",
    }

result = revenue_simulation()
for k, v in result.items():
    print(f"  {k}: {v}")
```

### Sensitivity Analysis

Identify which variables matter most:

```python
import numpy as np

def sensitivity_analysis():
    """
    One-at-a-time sensitivity analysis.
    Vary each input +/-20% and measure impact on output.
    """
    base_params = {
        "visitors": 50_000,
        "conversion_rate": 0.05,
        "avg_order_value": 50,
        "months": 12,
    }

    def revenue(**params):
        return params["visitors"] * params["conversion_rate"] * params["avg_order_value"] * params["months"]

    base_revenue = revenue(**base_params)
    print(f"Base revenue: ${base_revenue:,.0f}\n")

    for param_name, base_value in base_params.items():
        low_params = {**base_params, param_name: base_value * 0.8}
        high_params = {**base_params, param_name: base_value * 1.2}

        low_revenue = revenue(**low_params)
        high_revenue = revenue(**high_params)
        swing = high_revenue - low_revenue

        print(f"{param_name}:")
        print(f"  -20%: ${low_revenue:,.0f}  |  +20%: ${high_revenue:,.0f}  |  Swing: ${swing:,.0f}")

sensitivity_analysis()
```

---

## Part 5: Practical Python Examples

### Quick Data Exploration Template

```python
import pandas as pd

def explore(df: pd.DataFrame, target_col: str = None):
    """Quick exploration of a DataFrame for decision-making."""
    print("=" * 60)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print("=" * 60)

    print("\n--- Column Types ---")
    print(df.dtypes.value_counts())

    print("\n--- Missing Values (top 10) ---")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(10)
    if len(missing) > 0:
        for col, count in missing.items():
            print(f"  {col}: {count:,} ({count/len(df):.1%})")
    else:
        print("  No missing values")

    print("\n--- Numeric Summary ---")
    print(df.describe().round(2))

    if target_col and target_col in df.columns:
        print(f"\n--- Target: {target_col} ---")
        print(df[target_col].value_counts().head(10))
```

### A/B Test Analysis Pipeline

```python
def full_ab_analysis(control_data, treatment_data, metric_name="conversion"):
    """
    Complete A/B test analysis pipeline.
    Combines frequentist and Bayesian approaches.
    """
    # Frequentist analysis
    freq_result = analyze_ab_test(
        control_visitors=len(control_data),
        control_conversions=sum(control_data),
        treatment_visitors=len(treatment_data),
        treatment_conversions=sum(treatment_data),
    )

    # Bayesian analysis
    bayes_result = bayesian_ab_test(
        control_successes=sum(control_data),
        control_trials=len(control_data),
        treatment_successes=sum(treatment_data),
        treatment_trials=len(treatment_data),
    )

    print(f"=== A/B Test Report: {metric_name} ===\n")
    print("Frequentist Results:")
    for k, v in freq_result.items():
        print(f"  {k}: {v}")
    print("\nBayesian Results:")
    for k, v in bayes_result.items():
        print(f"  {k}: {v}")

    # Decision recommendation
    print("\n--- Recommendation ---")
    if freq_result["significant"] and float(bayes_result["prob_treatment_better"].rstrip('%')) > 95:
        print("SHIP IT: Strong evidence treatment is better.")
    elif float(bayes_result["prob_treatment_better"].rstrip('%')) > 80:
        print("LEAN SHIP: Moderate evidence. Consider business context.")
    else:
        print("HOLD: Insufficient evidence. Collect more data or iterate.")
```

---

## Part 6: Cognitive Biases in Data Interpretation

Even with good data, biases distort decisions:

| Bias | Description | Countermeasure |
|------|-------------|---------------|
| **Confirmation bias** | Seeking data that supports existing beliefs | Pre-register hypotheses; have a skeptic review |
| **Survivorship bias** | Analyzing only successes, ignoring failures | Include churned users, failed experiments |
| **Anchoring** | Over-weighting the first number seen | Generate multiple estimates independently |
| **Base rate neglect** | Ignoring how common something is overall | Always start with the base rate |
| **Availability bias** | Overweighting recent or dramatic events | Use systematic data, not anecdotes |
| **Simpson paradox** | Aggregate trend reverses in subgroups | Always segment before concluding |

---

## Part 7: Building a Data-Driven Culture

### Organizational Habits

1. **Default to evidence**: Proposals include data, not just opinions
2. **Pre-register hypotheses**: State what you expect to find before looking at data
3. **Share negative results**: Failed experiments are valuable information
4. **Maintain a decision log**: Record decisions, reasoning, and outcomes for review. Tools like [KeepRule](https://keeprule.com) help teams codify the decision principles they learn from past experiments into reusable guidelines.
5. **Distinguish opinions from data**: Use phrases like "I believe" vs "the data shows"

### The Data Maturity Ladder

| Level | Description | Characterized By |
|-------|-------------|-----------------|
| 1. Ad Hoc | Decisions based on gut feel | No tracking, no metrics |
| 2. Reporting | Dashboards exist | Backward-looking, descriptive |
| 3. Analysis | Questions are answered with data | Analysts investigate specific questions |
| 4. Experimentation | A/B testing is standard practice | Hypotheses are tested before shipping |
| 5. Prediction | ML models inform decisions | Forward-looking, prescriptive |

---

## Cheat Sheets

### Which Test to Use?

| Question | Recommended Test |
|----------|-----------------|
| Is A different from B? (proportions) | Chi-squared test or Z-test for proportions |
| Is A different from B? (continuous) | t-test (if normal) or Mann-Whitney U (if not) |
| Is there a trend over time? | Linear regression or Mann-Kendall test |
| Which of 3+ options is best? | ANOVA (if normal) or Kruskal-Wallis |
| Are two variables related? | Pearson (linear) or Spearman (monotonic) correlation |
| What is the probability B > A? | Bayesian A/B test |

### Sample Size Rules of Thumb

| Baseline Rate | Detect 5% Lift | Detect 10% Lift | Detect 20% Lift |
|--------------|----------------|-----------------|-----------------|
| 1% | 3.1M per variant | 780K | 196K |
| 5% | 121K | 31K | 8K |
| 10% | 57K | 14K | 4K |
| 20% | 25K | 6K | 2K |
| 50% | 6K | 2K | 400 |

---

## Further Resources

- [Trustworthy Online Controlled Experiments](https://experimentguide.com/) — Kohavi, Tang, Xu
- [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) — Cameron Davidson-Pilon
- [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow) — Daniel Kahneman
- [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) — Richard McElreath
- [KeepRule](https://keeprule.com) — Capture and structure the decision principles your team learns from data analysis

---

## Contributing

Contributions are welcome — especially real-world examples and additional Python utilities. Please open an issue before submitting large changes.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
