# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Developer**: Machine Learning DevOps Engineer
- **Model Type**: Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`)
- **Model Version**: 1.0.0
- **Release Date**: October 2026
- **Hyperparameters**: `n_estimators=100`, `max_depth=15`, `min_samples_split=10`, `random_state=42`
- **Preprocessing**: One-hot encoding on categorical features using `OneHotEncoder(sparse_output=False, handle_unknown="ignore")` and binary label encoding on the target using `LabelBinarizer`.

## Intended Use
- **Primary Intended Uses**: The model is intended to classify whether an individual's annual income exceeds \$50,000 (`>50K` vs `<=50K`) based on demographic and employment attributes extracted from census data.
- **Primary Intended Users**: Data scientists, economists, social policy researchers, and ML engineers studying economic disparities or evaluating production ML serving pipelines.
- **Out-of-Scope Use Cases**: This model is not intended or certified for making automated, legally binding decisions, including credit granting, mortgage qualification, insurance underwriting, or hiring screenings. Using this model for high-stakes individual decisions risks perpetuating historical biases.

## Training Data
- **Dataset**: Publicly available 1994 US Census Bureau database extracted by Barry Becker and hosted by the UCI Machine Learning Repository (often referred to as the Adult / Census Income dataset).
- **Training Partition**: 80% random split of the cleaned census dataset, comprising 26,048 observations.
- **Attributes**: 14 input features comprising continuous variables (`age`, `fnlgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and categorical variables (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).
- **Data Preprocessing**: Extraneous whitespace in column headers and string values was removed. Categorical features were one-hot encoded and the target variable `salary` was binarized.

## Evaluation Data
- **Evaluation Partition**: 20% holdout test split of the cleaned census dataset, comprising 6,514 observations.
- **Evaluation Methodology**: The test dataset was preprocessed using the exact transformer objects (`OneHotEncoder` and `LabelBinarizer`) fitted exclusively on the training partition to prevent data leakage.

## Metrics
- **Performance Measures**: The model was evaluated using Precision, Recall, and F1-Score (F-beta with $\beta = 1.0$) for the positive salary bracket (`>50K`).
- **Overall Performance on Test Set**:
  - **Precision**: 0.7875 (78.75% of individuals predicted to earn >\$50K actually earn >\$50K)
  - **Recall**: 0.5754 (57.54% of all individuals who earn >\$50K were correctly identified)
  - **F1-Score**: 0.6650 (Harmonic mean of precision and recall)
- **Slice Performance**:
  Model metrics were computed across slices for every unique value of all categorical features (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`). Complete per-slice metrics are persisted in `slice_output.txt`.
  - For example, within the `education` feature:
    - **Doctorate**: Count = 77, Precision = 0.8500, Recall = 0.8947, F1-Score = 0.8718
    - **Prof-school**: Count = 116, Precision = 0.8298, Recall = 0.9286, F1-Score = 0.8764
    - **Masters**: Count = 369, Precision = 0.8119, Recall = 0.8551, F1-Score = 0.8329
    - **Bachelors**: Count = 1053, Precision = 0.7444, Recall = 0.8089, F1-Score = 0.7753
    - **HS-grad**: Count = 2085, Precision = 0.9195, Recall = 0.2319, F1-Score = 0.3704

## Ethical Considerations
- **Demographic Bias**: The 1994 Census dataset exhibits historical gender, racial, and geographic wage disparities. Predictions reflect historical inequalities present in 1990s US labor markets rather than individual potential or merit.
- **Fairness Monitoring**: Model performance varies across demographic slices (e.g. sex, race, native country). Automated deployment without fairness safeguards could reinforce existing socioeconomic stratification.
- **Mitigation & Governance**: Ongoing algorithmic auditing and slice monitoring are essential. The model should never serve as the sole automated gatekeeper for resource distribution, employment, or financial services.

## Caveats and Recommendations
- **Temporal Relevance**: The dataset originates from 1994; inflation, structural shifts in industry, and evolving wage structures make the raw thresholds outdated for current economic prediction.
- **Class Imbalance**: The target distribution is imbalanced (~76% `<=50K` vs ~24% `>50K`). Depending on downstream application requirements, classification thresholds may be adjusted to balance precision against recall.
- **Recommendations**: For modern production deployment, retrain the architecture on contemporary census or socioeconomic survey data. Maintain automated CI/CD testing pipelines and model slice validation on every new training iteration.
