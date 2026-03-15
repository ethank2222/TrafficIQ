---
layout: default
title: Final Report
---

## Video
- ~3 mins
    - HARD MAX 4 mins
- Brief problem description:
    - include images, screenshots, etc
- Before and after training peformance (I can supply the videos for this - Erick)
- Include failure modes
    - Not sure how we'll do this?
    - Maybe talk about models undesirable behaviors
        - Over prioritizing clearing certain lanes
        - Switches somewhat too often (wasted time on yellow)
- Summary of Approach + Future Ideas
    - p.much what we did for our presentation :P

## Project Summary
**Description DELETE LATER**
Write a couple of paragraphs summarizing the motivation and goals of the project; yes, yet again, but a more comprehensive version. In particular, make sure that the problem you’re addressing is clearly defined here, and feel free to use visual aides (e.g. an image or two) to clarify the setup. Part of the evaluation will be on how well you are able to motivate what’s interesting and challenging about the problem, i.e. why it is not trivial, and why you need AI/ML algorithms to solve it.

## Evaluation
**Description DELETE LATER**
An important aspect of your project is evaluation. Be clear and precise about describing the evaluation setup, for both quantitative and qualitative results. Present the results to convince the reader of the effort that you’ve made to solve the problem, and to what extent you can claim that you succeeded. Use plots, charts, tables, screenshots, figures, etc. as appropriate. For each type of evaluation that you perform, you’ll likely need at least a paragraph or two (excluding figures etc.) to describe it.
### Qualitative Results
ADD VIDEO OF TRAINED MODEL 10 EPOCHS
ADD VIDEO OF TRAINED MODEL 100 EPOCHS
For qualitative assessment, we evaluated the model against three behavioral criteria:
- Fairness: No lane starvation
    - All approaches are served within a reasonable time window
- Stability: No excessive or rapid phase switching that wastes green time on yellow transitions
- Adaptivity: Phase decisions that respond to live traffic conditions rather than following a fixed pattern

Throughout development, we validated our model by reviewing recordings of it in actio, observing its responses to incoming traffic, current queue state, and active light phase.

These reviews surfaced several flaws in our environment setup. First, mean vehicle length proved largely redundant, as nearly all vehicles in the simulation share similar dimensions with negligible variation in how they influence model behavior. We removed it accordingly. Second, earlier model iterations showed little awareness of traffic flow, frequently producing stop-and-go light patterns rather than sustained lane clearance.
To address this, we introduced two new observation features: **Mean Speed** and **Occupancy**. The effect was immediate, rather than reacting to static queue snapshots, the agent began actively prioritizing lane throughput and avoiding unnecessary light changes.
ADD VIDEO OF TRAINED MODEL 100 EPOCHS


ADD VIDEO OF WEBSTER ALGORITHIM
We as well attempted to capture the behavior seen in more refined algorithims such as websters. 
### Quantitative Results

## Resources Used
**Description DELETE LATER**
Mention all the resources that you found useful in implementing your method, experiments, and analysis. This should include everything like paper references, code documentation, AI/ML libraries, source code that you used, StackOverflow, images you included that you didn’t create yourselves, and any other websites/links you found useful. You do not have to include every tiny (or commonplace) thing you used, but it is important to report the sources that are crucial to your project. One aspect that does need to be comprehensive is a description of any use you made of AI tools, including what the tool was, how you used it, and in what form it appears in your report.