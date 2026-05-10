## Limitations

This dashboard has limitations users should understand before interpreting any output.

**The data only includes reported breaches.** The model is trained on incidents reported to the ICO under UK GDPR mandatory breach notification requirements. Breaches that were not reported, whether because they went undetected, were below the reporting threshold, or were under-reported, are invisible to the model. The dashboard cannot tell you anything about unreported breaches.

**Definitions changed in April 2021.** The ICO changed its definitions of *Informal Action Taken* and *No Further Action* in April 2021. To avoid mixing two regulatory regimes, the model is trained on data from 2021 Q2 onwards only. Outputs reflect post-April-2021 ICO practice and should not be used to interpret older incidents.

**Minority outcomes are sparsely represented.** Roughly 83% of historical breaches result in *Informal Action Taken*, leaving *Investigation Pursued* and *No Further Action* with much less data for the model to learn from. The model's predictions for these less common outcomes are correspondingly less reliable. The Simulator surfaces this directly through per-class calibration indicators.

**The dashboard predicts patterns, not decisions.** The model identifies what regulatory outcomes have historically tended to follow breaches with similar characteristics. It does not know about the specific facts of any individual breach, the organisation involved, the responsiveness of the data controller or the policy environment at the time of reporting. ICO decisions reflect all of these and more. The dashboard is therefore a research and educational tool, not a predictor of any specific real-world regulatory outcome.

**Clusters show similarity, not predictive groups.** The clustering output groups breaches by their characteristics (data types involved, incident type, scale, reporting time). Statistical evaluation showed that these clusters do not meaningfully align with regulatory outcomes meaning that breaches similar in shape do not necessarily share decisions. Treat cluster information as "what does this kind of breach look like based on previous breaches?", the model isn't advanced enough to treat it as a prediction.

**The training data is retrospective and the dashboard does not retrain.** The model was trained on data published up to 2025 Q4. Newer regulatory practice, changing categorisation or shifts in breach patterns since training are not reflected.

**The dashboard is not a substitute for regulatory or legal advice.** It is a research artefact built for academic dissertation purposes. Operational decisions about real breaches require professional advice and engagement with the ICO directly.