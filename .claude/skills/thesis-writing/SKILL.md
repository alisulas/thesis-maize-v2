---
name: thesis-writing
description: Use this skill whenever drafting, editing, or structuring content for the IEEE Access journal paper or thesis document. Triggers include any mention of paper section drafting (introduction, related work, methodology, experiments, results, discussion, conclusion), academic writing, IEEE format, LaTeX, citations, BibTeX, abstracts, or response to reviewer comments. Use this skill before writing any paper section. Do NOT use for non-academic writing or general LaTeX questions unrelated to the thesis paper.
---

# Thesis & Paper Writing Skill

This skill covers patterns for writing the IEEE Access journal paper for the maize yield transfer learning thesis.

## Critical Boundaries

**This skill does NOT generate full paper text autonomously.** Paper writing requires the researcher's authentic voice. This skill provides:
- Structural templates
- Style guidelines
- Citation patterns
- Specific phrasings to revise/improve drafts the researcher writes

The researcher MUST write the first draft of every section. Claude Code helps **revise, structure, and polish** — not generate from scratch.

## IEEE Access Paper Structure

Target length: 12-15 pages, two-column IEEE Access template.

### Standard Section Allocation

| Section | Pages | Word Count |
|---------|-------|------------|
| Abstract | 0.25 | 150-200 |
| Introduction | 1.5-2 | 800-1000 |
| Related Work | 1-1.5 | 600-800 |
| Methodology | 2.5-3 | 1500-2000 |
| Experiments | 1.5-2 | 800-1000 |
| Results | 3-4 | 1500-2000 |
| Discussion | 1-1.5 | 600-800 |
| Conclusion | 0.5 | 200-300 |
| References | 1-2 | 30-50 entries |

## Section-Specific Templates

### Abstract (write LAST, after results are finalized)

Structure (1 sentence each):
1. **Context**: Crop yield prediction is critical for food security.
2. **Gap**: Existing models trained on data-rich regions may not transfer to data-limited regions.
3. **Approach**: We investigate transfer learning from US county-level data to ASEAN province-level data for maize yield.
4. **Method**: We compare from-scratch baselines, vanilla fine-tuning, and DANN using MODIS satellite data, 2003-2023.
5. **Results**: We achieve [X% improvement / specific findings about domain gap].
6. **Significance**: Findings inform applicability of transfer learning across climate zones.
7. **Keywords**: 5-7 keywords (transfer learning, deep learning, crop yield prediction, remote sensing, MODIS, domain adaptation, ASEAN).

### Introduction Pattern

Three paragraphs, narrowing focus:

**Paragraph 1 — Broad Importance** (5-6 sentences):
- Global food security and climate context
- Role of accurate yield prediction
- Why ASEAN matters (population, food production, climate vulnerability)

**Paragraph 2 — Technical Gap** (5-6 sentences):
- Deep learning success in data-rich regions (USA: cite You 2017, Wang 2018)
- Challenge: data-limited regions have coarse spatial resolution and small sample sizes
- Transfer learning as potential solution
- Open question: does USA-trained model transfer effectively to tropical ASEAN?

**Paragraph 3 — Contribution Statement** (must be explicit):
> "In this paper, we make the following contributions:
> 1. We present the first systematic evaluation of transfer learning from US to ASEAN countries (Indonesia, Vietnam, Thailand) for maize yield prediction.
> 2. We compare vanilla fine-tuning against domain adversarial training (DANN) and analyze the impact of climate domain gap.
> 3. We provide ablation studies showing [specific finding about target sample size requirements].
> 4. Our open-source implementation and processed dataset enable reproducibility and extension to other crops/regions."

### Related Work Categorization

Organize by 3 themes (NOT chronologically):

1. **Deep Learning for Crop Yield Prediction**: You et al. 2017, Khaki et al. 2020/2021, Jiang et al. 2020
2. **Transfer Learning in Remote Sensing**: Wang et al. 2018, Schwalbert 2020, Zhao et al. 2022, Zhang et al. 2025
3. **Crop Yield Prediction in Tropical Regions**: Tropical-specific challenges, ASEAN-specific work (limited literature — emphasize this gap)

End with: "However, no prior work has systematically evaluated US-to-ASEAN transfer for maize, addressing the climate domain gap. Our work fills this gap."

### Methodology Section Template

Subsections (in order):

1. **Problem Formulation** (mathematical notation)
2. **Data Sources** (table with countries, sources, years)
3. **Feature Engineering** (histogram method from You 2017)
4. **Model Architecture** (figure showing CNN-LSTM)
5. **Transfer Learning Strategies** (subsections per method: vanilla, frozen, DANN)
6. **Training Procedure** (hyperparameters, optimizer, learning rate schedule)
7. **Evaluation Protocol** (metrics: RMSE, R², MAPE; train/val/test split rationale)

### Results Section Pattern

Lead with the **headline finding**, then drill down:

**Subsection 1**: USA Baseline (establish source model quality)
**Subsection 2**: ASEAN From-Scratch Baselines (establish target difficulty)
**Subsection 3**: Transfer Learning Performance (main results table)
**Subsection 4**: Per-Country Analysis (which country benefits most/least)
**Subsection 5**: Ablation Studies (sample size, domain adaptation strength)
**Subsection 6**: Failure Cases & Limitations

### Discussion Section Pattern

Don't repeat results. Address:

1. **Why did transfer learning work / not work?** (climate gap analysis)
2. **What does this mean for practitioners?** (when to use transfer learning)
3. **Comparison with prior work** (Wang 2018 cross-country soybean: similarities/differences)
4. **Limitations** (be honest: MODIS resolution, no in-season prediction, no genotype info)
5. **Future work** (multi-source, Brazil pretraining, attention mechanisms)

## Academic Writing Style Guidelines

### Voice
- **Active voice preferred**: "We propose..." not "It is proposed that..."
- **First person plural ("we")** is standard for IEEE papers
- **Present tense for general truths**: "Transfer learning leverages..."
- **Past tense for specific experiments**: "We trained the model..."

### Tone
- **Hedge claims**: "Our results suggest..." not "We prove..."
- **Quantify when possible**: "improves R² by 23%" not "significantly improves"
- **Acknowledge limitations explicitly**: "A limitation of this approach is..."

### Common Phrasings to Use

- "We hypothesize that..."
- "To the best of our knowledge..."
- "In contrast to prior work..."
- "Building on the framework of [author], we extend..."
- "Despite this success, [limitation]..."
- "These findings indicate..."
- "An important caveat is..."

### Phrasings to AVOID

- ❌ "It is well known that..." (cite or remove)
- ❌ "Obviously..." (if obvious, no need to state)
- ❌ "Very", "really", "extremely" (weak modifiers)
- ❌ "In conclusion..." (be more specific)
- ❌ "This paper investigates..." (use active: "We investigate...")

## Citation Patterns

### BibTeX Entry Format

```bibtex
@article{you2017deep,
  title={Deep Gaussian Process for Crop Yield Prediction Based on Remote Sensing Data},
  author={You, Jiaxuan and Li, Xiaocheng and Low, Melvin and Lobell, David and Ermon, Stefano},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={31},
  number={1},
  year={2017}
}
```

### Citation Style Rules

- **First mention of author**: "You et al. [3] proposed..."
- **Subsequent mentions**: "[3]" or "the method of [3]"
- **Multiple citations**: "[3, 5, 7]" or "[3]-[7]" for ranges
- **Key references**: Bold the foundational ones (You 2017, Wang 2018) in your mental model

### Reference Quality Hierarchy (cite preferentially)

1. **Top-tier conferences**: NeurIPS, ICML, ICLR, AAAI, IJCAI, KDD, CVPR
2. **Top journals**: Nature Sci. Rep., Remote Sensing of Environment, IEEE TGRS
3. **Crop science journals**: Field Crops Research, Agricultural Systems
4. **Reputable secondary**: arXiv preprints with high citations
5. **Avoid**: Non-peer-reviewed blogs, Wikipedia, MDPI predatory journals (some Remote Sensing MDPI is fine, but check)

## Figure & Table Standards

### Figures
- Vector format (PDF/SVG) preferred over raster
- Font: Helvetica or sans-serif, 10pt minimum
- Color scheme: Colorblind-safe (use viridis, not red-green)
- Caption format: "Fig. X. [Description]. [Additional context if needed.]"
- Place near first reference in text

### Tables
- IEEE format: horizontal lines only (no vertical)
- Caption format: "TABLE X: [Description]"
- Bold the best result in each column
- Always include units in column headers
- Avoid abbreviations in table without footnote

### Standard Tables for This Paper

**Table 1**: Dataset summary (country, source, years, n_samples)
**Table 2**: Hyperparameters
**Table 3**: Main results (rows: methods, columns: countries × metrics)
**Table 4**: Ablation results

## LaTeX Snippets

### IEEE Access Article Template Header

```latex
\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08emT\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}
\title{Transfer Learning for Maize Yield Prediction from Data-Rich to Data-Limited Environments: A Case Study of ASEAN Countries}
\author{[Author Name], \IEEEmembership{Member, IEEE}}
\maketitle

\begin{abstract}
[Abstract text]
\end{abstract}

\begin{IEEEkeywords}
Transfer learning, deep learning, crop yield prediction, remote sensing, MODIS, domain adaptation, Southeast Asia
\end{IEEEkeywords}

\section{Introduction}
[Content]

% ... other sections ...

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

### Equation Standards

```latex
% Inline equation
The yield prediction is $\hat{y} = f_\theta(\mathbf{x})$ where $\mathbf{x}$ is the histogram input.

% Display equation with label
\begin{equation}
\mathcal{L}_{\text{DANN}} = \mathcal{L}_y - \lambda \mathcal{L}_d
\label{eq:dann_loss}
\end{equation}

% Reference to equation
As shown in Equation~\ref{eq:dann_loss}, ...
```

## Workflow: Iterating on a Section

When working on any section with Claude Code, follow this workflow:

1. **You write first draft** in `paper/sections/[section_name].tex` (rough, no polish)
2. **Ask Claude Code**: "Review my draft of [section]. Identify: unclear sentences, missing citations, weak transitions, redundancies. Don't rewrite — just point out issues."
3. **You revise** based on feedback
4. **Ask Claude Code**: "Suggest 3 alternative phrasings for [specific sentence]" (only when stuck)
5. **You make final choice**
6. **Ask Claude Code**: "Check this paragraph for grammar, IEEE style consistency, and citation format" (final polish)

**Anti-pattern**: "Write the introduction for me" → results in generic, detectable AI text. Reviewers and supervisors will notice.

## Reviewer Response Template

When responding to reviewers (after first submission):
Reviewer 1, Comment 1:
[Quote reviewer comment]
Response:
We thank the reviewer for this insightful comment. [Acknowledge validity]. To address this concern, we have:

[Specific change made]
[Additional analysis added]
The revised text appears in Section [X], page [Y], lines [Z-W]. [If applicable: We also added Table/Figure [N].]


Be polite, specific, and never argumentative even if reviewer misunderstood.

## Critical Rules

1. **No fabricated citations**. EVER. Verify every reference exists and the claim is accurate.
2. **No fabricated numbers**. If results aren't in yet, use placeholder `[X.XX]` and fill later.
3. **Track changes when revising**. Use git for version control of paper/.
4. **Respect supervisor authorship norms**. Add supervisor as co-author from draft 1.
5. **Plagiarism check before submit**. Use Turnitin or iThenticate via your university.

## Validation Checklist Before Submission

- [ ] All citations have BibTeX entries that resolve
- [ ] All figures rendered correctly in PDF
- [ ] All tables fit within column width
- [ ] Abstract under 200 words
- [ ] Keywords listed (5-7)
- [ ] Acknowledgments section present
- [ ] Conflict of interest declaration
- [ ] Plagiarism check passed (<15% similarity)
- [ ] Spell-check passed
- [ ] Supervisor approved final version
- [ ] Compiled PDF reviewed page-by-page

## Related Files in Project

- Paper source: `paper/main.tex`
- Sections: `paper/sections/*.tex`
- Bibliography: `paper/references.bib`
- Figures: `paper/figures/*.pdf`
- Notes: Obsidian vault (separate from project)

## When NOT to Use This Skill

- General writing assistance (this is for academic IEEE Access only)
- Code documentation (use docstrings instead)
- README writing (use README.md template)
- Email/communication drafting
