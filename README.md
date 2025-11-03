# LA Crime Data Analysis and Prediction

## Problem Description

Crime prediction is essential for effective law enforcement and public safety. This project addresses the challenge of classifying crimes in Los Angeles as either violent or non-violent based on various incident characteristics such as location, time, victim demographics, and crime details. Accurate prediction enables law enforcement agencies to allocate resources efficiently, implement preventive measures in high-risk areas, and improve emergency response planning.

Using crime data from 2020 to present, we developed and compared four machine learning models: Decision Tree, Naive Bayes, Random Forest, and K-Nearest Neighbors. Our analysis demonstrates that Random Forest achieves the highest accuracy while providing valuable insights into which features most strongly predict crime severity. The project reveals that weapon type, premise location, and time of day are the strongest predictors of violent crime.

---

## Dataset Source

**Source:** Los Angeles Police Department Crime Data (2020-Present)

**Dataset Size:**
- Hundreds of thousands of crime records spanning multiple years
- 28 original columns with temporal, spatial, victim, and incident details
- Final dataset after preprocessing and outlier removal

**Key Features:**

| Feature | Type | Description |
|---------|------|-------------|
| DATE OCC | Datetime | Date when crime occurred |
| TIME OCC | Integer | Time when crime occurred (military format) |
| AREA | Categorical | Geographic area code (1-21) |
| AREA NAME | Categorical | Name of geographic area |
| Crm Cd Desc | Text | Detailed crime description |
| Vict Age | Numerical | Age of victim |
| Premis Cd | Categorical | Premise type code |
| Premis Desc | Text | Description of premise type |
| Weapon Used Cd | Categorical | Weapon code if applicable |
| LAT/LON | Numerical | Geographic coordinates |

**Preprocessing Steps:**

1. **Date and Time Extraction:**
   - Converted DATE OCC to datetime format
   - Extracted year, month, and hour components for temporal analysis
   - Removed original date columns to reduce dimensionality

2. **Crime Categorization:**
   Manually classified crimes into three categories based on crime descriptions:
   
   - **Violent crimes:** Battery, assault, homicide, robbery, rape, kidnapping, weapon-related incidents
   - **Theft-related:** Burglary, shoplifting, identity theft, vandalism, petty theft, grand theft
   - **Vehicular crimes:** Vehicle theft, burglary from vehicle, motor vehicle theft
   
   Created binary target variable `crime_type` (violent vs nonviolent)

3. **Outlier Handling:**
   Applied Interquartile Range (IQR) method to remove extreme values in victim age:
   - Q1 = 25th percentile, Q3 = 75th percentile
   - IQR = Q3 - Q1
   - Removed values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
   - This eliminated unrealistic ages (negative values, ages over 100)

4. **Feature Encoding:**
   Converted categorical variables to numerical format using Label Encoding for:
   - AREA (geographic regions)
   - Premis Cd (premise types)
   - Weapon Used Cd (weapon categories)

5. **Train-Test Split:**
   - 70% training set, 30% testing set
   - Stratified sampling to maintain class distribution
   - Random state = 42 for reproducibility

**Exploratory Insights:**

<img width="1427" height="701" alt="Image" src="https://github.com/user-attachments/assets/ba902d49-8305-4c9b-bfae-69029aa6ef51" />

The most common crimes are battery/assault and vehicle-related theft, with property crimes significantly outnumbering violent crimes.

<img width="1023" height="774" alt="Image" src="https://github.com/user-attachments/assets/6cae4d92-cf3d-40e0-b5ea-48829285624d" />

Crime patterns vary substantially across LA neighborhoods, with central and downtown areas showing more diverse crime types.

<img width="1333" height="701" alt="image" src="https://github.com/user-attachments/assets/6746ba7c-085c-48f2-a6d4-b182bf3df1e6" />

Streets are the most common crime location, followed by single-family residences and parking lots.

---

## Methods

### Feature Selection

Based on exploratory analysis and domain knowledge, we selected six features most predictive of crime type:

| Feature | Rationale |
|---------|-----------|
| AREA | Different neighborhoods have distinct crime patterns |
| hour | Time of day correlates with crime type (e.g., nighttime assaults) |
| month | Seasonal patterns in crime occurrence |
| Vict Age | Victim demographics can indicate crime type |
| Premis Cd | Location type is strongly predictive (street vs residence vs commercial) |
| Weapon Used Cd | Strong indicator of violent vs non-violent crime |

We considered using additional features like victim sex, reporting district, and geographic coordinates but found these six provided the best balance of predictive power and model interpretability.

### Model Selection and Rationale

We implemented four machine learning algorithms, each chosen for different strengths:

#### 1. Decision Tree Classifier

**Why Decision Trees:**
- Provides interpretable rule-based classifications
- Handles both numerical and categorical features without scaling
- Shows clear decision paths
- Serves as strong baseline for comparison

**Configuration:**
- Default parameters initially (no max depth)
- Gini impurity criterion for splitting
- Used PCA for 2D visualization of decision boundary

<img width="870" height="722" alt="image" src="https://github.com/user-attachments/assets/b432025d-8ffe-4664-ba31-2bce335d04f1" />

**Trade-offs:**
- Pros: Fast, interpretable, no preprocessing needed
- Cons: Prone to overfitting, unstable with small data changes

---

#### 2. Naive Bayes Classifier

**Why Naive Bayes:**
- Extremely fast training and prediction
- Works well with mixed numerical and categorical features
- Provides probabilistic predictions useful for confidence estimation
- Good baseline despite independence assumption

**Configuration:**
- Gaussian Naive Bayes for numerical features (hour, month, Vict Age)
- Categorical Naive Bayes for categorical features (AREA, Premis Cd, Weapon Used Cd)
- Combined predictions through probability averaging

**Alternative Considered:** Using only Gaussian NB on all features after encoding, but hybrid approach performed better by respecting feature types.

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/47ce96e6-3bbc-49f7-a7e0-f95848d62468" />

**Trade-offs:**
- Pros: Very fast, handles missing data well, probabilistic outputs
- Cons: Independence assumption often violated, can be outperformed

---

#### 3. Random Forest Classifier

**Why Random Forest:**
- Ensemble method reduces overfitting compared to single decision tree
- Provides feature importance rankings
- Generally achieves highest accuracy
- Robust to hyperparameter choices

**Configuration:**
- 50 trees (n_estimators=50)
- Gini criterion for splitting
- Max depth of 25 to prevent overfitting
- Max features of 4 per split (introduces randomness)
- Class weight balanced to handle imbalanced data

**Alternative Considered:** Gradient Boosting (XGBoost), but Random Forest provided similar accuracy with better interpretability and faster training.

<img width="643" height="455" alt="image" src="https://github.com/user-attachments/assets/72618f2c-6736-4da4-b0b9-c31d15e8817e" />

The feature importance analysis reveals:
1. Weapon Used Cd is the strongest predictor (violent crimes typically involve weapons)
2. Premise Code is second (location type matters significantly)
3. Hour and Area show moderate importance
4. Month and Victim Age have lesser but still meaningful impact

<img width="567" height="435" alt="image" src="https://github.com/user-attachments/assets/fe279328-7364-43ac-944f-f999656b9ad5" />

**Trade-offs:**
- Pros: High accuracy, feature importance, resistant to overfitting
- Cons: Slower than simple models, less interpretable than single tree, requires more memory

---

#### 4. K-Nearest Neighbors (KNN)

**Why KNN:**
- No explicit training phase (instance-based learning)
- Naturally handles non-linear decision boundaries
- Simple and intuitive approach
- No assumptions about data distribution

**Configuration:**
- k=11 neighbors (optimized through error rate analysis)
- Mean imputation for missing values
- Euclidean distance metric

**Hyperparameter Optimization:**

We tested k values from 1 to 20 to find the optimal number of neighbors:

<img width="691" height="455" alt="image" src="https://github.com/user-attachments/assets/fe7e47ee-a0f4-40f1-b638-77da3c1473ce" />

The error rate analysis shows:
- k=1 overfits (memorizes training data)
- k=11 minimizes test error
- Higher k values increase bias, underfit the data

<img width="559" height="455" alt="image" src="https://github.com/user-attachments/assets/36470827-cc74-456e-8137-935d24db6542" />

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/bd3d66f3-1734-4b70-932e-110988660b81" />


**Trade-offs:**
- Pros: Simple, no training phase, handles non-linear patterns
- Cons: Slow prediction with large datasets, sensitive to feature scaling, memory-intensive

---

### Model Comparison Framework

We compared models using multiple metrics to get a complete picture:

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall correctness across both classes |
| Precision | How many predicted violent crimes are actually violent |
| Recall | How many actual violent crimes were caught |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Overall discrimination ability across thresholds |
| Confusion Matrix | Detailed breakdown of prediction types |

---

## Steps to Run the Code

### Prerequisites

Install required Python libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Data Preparation

1. Download the dataset from LA Open Data Portal
2. Place `Crime_Data_from_2020_to_Present.csv` in a `data/` folder
3. Ensure the file path matches the code: `data/Crime_Data_from_2020_to_Present.csv`

### Execution Steps

**Step 1: Load and preprocess data**
- Run the data loading and preprocessing cells
- This creates cleaned features and target variable
- Handles missing values and outliers
- Performs train-test split

**Step 2: Train Decision Tree**
- Fits Decision Tree on training data
- Generates predictions and performance metrics
- Creates decision boundary visualization using PCA

**Step 3: Train Naive Bayes**
- Trains Gaussian NB on numerical features
- Trains Categorical NB on categorical features
- Combines predictions and evaluates performance
- Plots ROC curve

**Step 4: Train Random Forest**
- Fits Random Forest with specified hyperparameters
- Calculates feature importance
- Evaluates on both training and test sets
- Generates accuracy comparison plot

**Step 5: Optimize and train KNN**
- Tests different k values (1-20)
- Selects optimal k based on error rate
- Trains final KNN model
- Creates confusion matrix and precision-recall curve

**Step 6: Compare all models**
- Generates accuracy comparison bar chart
- Creates confusion matrices for all four models
- Plots ROC curves on the same graph
- Produces summary performance table

### Expected Output

After running all cells, you'll have:
- Trained model objects for each algorithm
- Performance metrics printed to console
- Multiple visualization plots showing model comparison
- Feature importance rankings (from Random Forest)
- Understanding of which model works best for this problem

---

## Experiments and Results

### Overall Performance Comparison

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/af900d50-06dd-4266-a86e-5e66dec001f6" />

**Performance Summary Table:**

| Model | Accuracy | Training Speed | Prediction Speed |
|-------|----------|----------------|------------------|
| Random Forest | Highest | Moderate | Moderate |
| KNN | Second Best | Instant (no training) | Slow |
| Decision Tree | Good | Fast | Fast |
| Naive Bayes | Competitive | Very Fast | Very Fast |

Random Forest achieves the best accuracy, followed closely by KNN. Decision Tree and Naive Bayes provide competitive performance with faster computation times.

### Confusion Matrix Analysis

<img width="1389" height="990" alt="image" src="https://github.com/user-attachments/assets/f07df33d-c0d6-43fc-8840-bec4e2240f79" />

**Key Observations:**

**Random Forest:**
- Best balance between True Positives and True Negatives
- Lowest false negative rate (misses fewest violent crimes)
- Slightly higher false positives but acceptable

**KNN:**
- Strong performance overall
- More false negatives than Random Forest
- Fewer false positives show it's conservative in predicting violent

**Decision Tree:**
- Moderate performance
- Shows signs of overfitting (better on training than test)
- More balanced but lower overall accuracy

**Naive Bayes:**
- Higher false positive rate
- Independence assumption limits performance
- Fast but less accurate than ensemble methods

### ROC Curve Comparison

<img width="846" height="626" alt="image" src="https://github.com/user-attachments/assets/fc2e89a8-be07-4b7f-8d80-40f5e62d7f1a" />

The ROC curves illustrate each model's ability to distinguish between violent and non-violent crimes across different classification thresholds:

- **Random Forest** achieves the highest AUC, indicating best overall discrimination
- **KNN** follows closely with strong true positive rate
- **Decision Tree** shows good but not exceptional performance
- **Naive Bayes** has the lowest AUC but still performs above random chance

The diagonal line represents random guessing (AUC = 0.5). All models significantly outperform random classification.

### Feature Importance Insights

From Random Forest analysis:

1. **Weapon Used Cd (highest importance)** - Makes intuitive sense as violent crimes typically involve weapons
2. **Premise Cd** - Location type strongly correlates with crime type
3. **Hour** - Time of day patterns differ between violent and non-violent crimes
4. **AREA** - Geographic patterns exist in crime types
5. **Vict Age** - Demographics play a role but less strongly
6. **Month** - Seasonal patterns have minimal but measurable impact

### Hyperparameter Analysis

**Random Forest Tree Count:**
- Tested 10, 25, 50, 100, 200 trees
- 50 trees provided optimal balance between accuracy and computation time
- Beyond 50 trees showed diminishing returns

**KNN Neighbor Count:**
- k=1 severely overfits (100% training accuracy, poor test performance)
- k=11 minimizes test error
- k>15 begins to underfit as model becomes too simple

---

## Conclusion

### Key Results

1. **Random Forest is the best performer** with highest accuracy and best balance between precision and recall, making it the recommended choice for deployment.

2. **Feature importance analysis** reveals weapon type and location are the strongest predictors of crime severity, which aligns with domain knowledge and provides actionable insights for law enforcement.

3. **Time of day matters significantly** - certain hours show higher rates of violent crime, suggesting targeted patrol strategies could be effective.

4. **Geographic patterns are strong** - different LA areas have distinct crime profiles, supporting neighborhood-specific prevention strategies.

5. **KNN performs well** as a simpler alternative, offering good accuracy with faster training, though slower prediction makes it less suitable for real-time systems.

### What We Learned

**About the Data:**
- Class imbalance exists (more non-violent crimes) but is manageable with proper techniques
- Outliers in victim age required careful handling
- Temporal and spatial features are highly predictive

**About Model Selection:**
- Ensemble methods (Random Forest) outperform single models
- Simple models (Naive Bayes, Decision Tree) provide good baselines and faster computation
- Instance-based learning (KNN) works well for complex patterns but scales poorly

**About Crime Prediction:**
- Violence is predictable from relatively few features
- No single feature dominates; combination of factors determines crime type
- Model interpretability (feature importance) is valuable for real-world application

### Practical Applications

This analysis can help law enforcement:
- **Allocate resources** based on predicted crime severity in different areas and times
- **Prioritize responses** to calls likely involving violent situations
- **Design prevention programs** targeting high-risk locations and times
- **Optimize patrol schedules** based on temporal crime patterns

---

## References

**Dataset:**
- Los Angeles Police Department. Crime Data from 2020 to Present. LA Open Data Portal.

**Libraries and Frameworks:**
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering.
- Waskom, M. (2021). Seaborn: Statistical data visualization. JOSS.

**Machine Learning Methods:**
- Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
- Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory.
- Quinlan, J. R. (1986). Induction of decision trees. Machine Learning, 1(1), 81-106.
