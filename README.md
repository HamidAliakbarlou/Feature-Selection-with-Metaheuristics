# Feature Selection with Metaheuristics (Genetic Algorithm, SA, VNS)

### Python Implementation of Metaheuristic Feature Selection for Enhanced Credit Granting"

Feature selection is imperative in machine learning, improving computational efficiency and predictive accuracy by isolating relevant predictors. However, the task is complex, being NP-hard and lacking polynomial time solutions. To address this challenge, our project explores modern metaheuristic methods, including Simulated Annealing, Variable Neighborhood Search, and Genetic Algorithm, applied to feature selection tasks using a Support Vector Machine model. These approaches offer diverse strategies to streamline feature selection processes and enhance model efficacy.

Our study showcases significant advancements in prediction accuracy compared to the base model with all features activated. Notably, the SVM model achieved a substantial increase in validation set accuracy, from 0.70625 to 0.742, representing an improvement of approximately 5%. Despite a slight decrease in accuracy on the test set, the models demonstrated robust generalization capabilities on unseen data, laying the groundwork for future scalability and exploration with larger datasets.

Furthermore, feature selection comparison sheds light on the effectiveness of different metaheuristic methods. Simulated Annealing, Variable Neighborhood Search, and Genetic Algorithm outperformed the base model and showcased competitive performance. Simulated Annealing, particularly, exhibited superior efficacy compared to Variable Neighborhood Search and demonstrated similar effectiveness to Genetic Algorithm. This highlights the potential of metaheuristic methods in feature selection tasks and underscores opportunities for further research to optimize predictive modeling in credit granting scenarios.

## Dataset
The dataset we used for this project is the German Credit Data, which categorises loan applicants
as good or bad credit risks based on a set of attributes. There are 1000 observations in this
dataset, with 21 variables (7 numerical and 13 categorical), and no missing values. The target
variable Y is a binary variable with two levels: good credit and bad credit. If a loan applicant is
labelled as 0, it indicates that he or she has "bad credit" based on the 21 variables associated, or
vice versa. Each variable is described in detail in the appendix (Table 1: Description of the
dataset).

Granting credit plays a key role in financial transactions. This task typically involves a lot of borrower-related variables. However, in many situations, having a large number of variables can introduce noise into the database when building predictive models. In this context, feature selection presents itself as a way to simplify a database by identifying key features, reducing computational costs, and improving prediction performance.

