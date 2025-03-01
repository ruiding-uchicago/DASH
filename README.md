**This repository supports our \*Science Advances\* paper, “Leveraging Data Mining, Active Learning, and Domain Adaptation for Efficient Discovery of Advanced Oxygen Evolution Electrocatalysts,” (previous preprint on arXiv <https://arxiv.org/abs/2407.04877>) and serves as a roadmap to our data and code. The repository is organized into two main parts: Experimental Records, Raw Data, Supplementary Notes, and Figures (released under CC0 or CC-BY) and the Machine Learning Scripts (released under the MIT License). All materials are available on Dryad (DOI: 10.5061/dryad.nk98sf83g) and GitHub (<https://github.com/ruiding-uchicago/DASH>).**

**Experimental Records, Raw Data, and Supplementary Notes and Figures**

**Overview:**  
This repository contains experimental data—including raw electrochemical measurements from all active learning trials—as well as comprehensive records and supplementary notes (both figures and detailed technical notes). These files not only document the data underlying the paper’s conclusions but also serve as a resource for reproducibility and further reanalysis.

**1\. Batch Folders (batch0 through batch5)**

Each batch folder corresponds to a distinct iteration of the active learning (AL) process. They collectively archive the raw electrochemical polarization curves from a total of 258 experimental trials. These data files are provided in both binary (.bin) and plain text (.txt) formats, and filenames are consistent with the identifiers used in the master record file.

**Structure within each batch folder:**

- **batch0:**
  - **Purpose:** Contains baseline electrochemical measurements collected before the ML-guided active learning iterations began.
  - **Data:** Raw polarization curve files in both .bin and .txt formats.
- **batch1 to batch5:**
  - **Each batch folder (batch1, batch2, batch3, batch4, batch5) is further divided into two subfolders:**
    - **final400:**
      - **Purpose:** Contains all raw measurement files from experiments that used an annealing temperature of 400°C.
      - **Data Format:** Includes both .bin and .txt files; filenames match sample identifiers from the master spreadsheet.
    - **final500:**
      - **Purpose:** Contains corresponding raw data for experiments conducted at an annealing temperature of 500°C.
      - **Data Format:** As with final400, the files are provided in .bin and .txt formats.

**Key Points:**

- All files in these folders are named according to the sample identifiers detailed in the “AL Full Experimental Records.xlsx” file.
- These data form the quantitative experimental foundation for the AL-guided catalyst discovery, capturing the evolution from baseline to ML-refined formulations.

**2\. AL Full Experimental Records.xlsx**

**Purpose and Content:**

- This Excel spreadsheet is the master record that compiles detailed data from all 258 experimental trials across batches 0–5.
- **Key Columns Include:**
  - **Sample Name:** Unique identifier for each trial, which directly corresponds to filenames in the batch folders.
  - **Element Compositions:** Detailed quaternary (or multi-metal) formulations used in each experiment.
  - **Synthesis Parameters:** Precise conditions including hydrothermal settings, annealing temperature (400 or 500°C), annealing duration, and any post-processing steps (e.g., acid washing).
  - **Electrochemical Performance:** Measured η₁₀ values (overpotential at 10 mA cm⁻²) that quantify catalyst activity.
- **Usage:**
  - Acts as a central reference that links each experimental trial’s formulation and processing conditions to its measured performance.
  - Serves as a key resource for any reanalysis or replication of the AL process.

**3\. Online Repository Figures**

**Purpose:**  
This folder contains all supplementary visualizations referenced throughout the manuscript and supplementary notes. These images provide insight into both the data analysis and the modeling processes used in the study.

**Organization and File Naming Convention:**

- **Two Main Categories:**
    1. **Correlation Analyses (Figures OR1–OR12):**
        - **Content:**
            - Visualizations (e.g., heatmaps, scatter plots) showing correlation analyses between various variables using methods such as Kendall, Spearman, and Pearson correlations.
            - These figures support the commentary in Table S2 and help validate the diversity of features included in the ML models.
    2. **Supervised Data Mining Results (Figures OR17–OR120):**
        - **Content:**
            - Detailed outputs from the supervised ML data mining and model interpretation processes (e.g., SHAP values, partial dependence plots, clustering bar plots).
            - These figures provide in-depth insights into feature importance, model behavior, and data-driven decision-making within the ML workflow.
- **File Naming Convention:**  
    Each file name is structured to include several key components for immediate contextual understanding:
    1. **Figure Number:** For instance, “Figure OR55” indicates its sequential position.
    2. **Algorithm Abbreviation:** (e.g., “CAT”, “XGB”, “LGB”) indicating the ML algorithm used (such as CatBoost, XGBoost, or LightGBM).
    3. **Interpretation Method:** (e.g., “SHAP” or “PDP”) specifying the method used to derive model insights.
    4. **Visualization Type:** (e.g., “Summary&Clustering Bar Plots”, “Heatmap&Cohort Plots”, “H statistic”) that describes the style and purpose of the figure.
    5. **Feature Focus:** (e.g., “Element”, “Synthesis&Test”, or “All”) which identifies the subset of features analyzed.
    6. **Modeling Target:** (e.g., “Activity” or “Stability”) indicating whether the analysis is for OER activity or stability.
    7. **Dataset Source:** (e.g., “High-Quality” or “Full”) specifying which dataset was used for the analysis.

**Example:**

- 1. _"Figure OR55 CAT SHAP Summary&Clustering Bar Plots-Element-Activity-High-Quality.jpg"_  
        This file is a visualization created using the CatBoost algorithm, interpreted via SHAP analysis, and presented as summary and clustering bar plots. It focuses on element-related features for predicting OER activity, and the analysis is based on the high-quality dataset.
- **Supplementary Figures Usage:**
    1. These figures are integral for readers who wish to explore the analytical details and reproduce the data interpretation steps.
    2. They serve as visual proof of the data correlations, model interpretations, and other analytical processes that underpin the manuscript’s conclusions.

**4\. Supplementary Notes (Embedded within this Directory)**

**Purpose:**  
These documents provide extensive technical details beyond what is included in the main manuscript. They are intended for users who require a deeper understanding of the methodologies and computational approaches used in the study.

**Supplementary Note Overviews:**

- **Supplementary Note 1: Details of Domain Knowledge Dataset**
  - Describes the literature search strategy, the criteria for high-quality data selection, and the imputation techniques used for missing values.
  - Explains how raw experimental variables (e.g., precursor compositions, synthesis conditions) were digitized and enriched with atomic properties.
- **Supplementary Note 2: Details of Unsupervised Data Mining**
  - Provides a comprehensive description of methods such as the Bibliometric Interconnected Network Graph and Apriori associate rule mining.
  - Includes additional visualizations and interpretation of element associations and stability factors that informed the focus on Ru-based catalysts.
- **Supplementary Note 3: Details of Dimensional Reduction Analysis**
  - Explains the application of PCA and t-SNE to the high-dimensional dataset.
  - Describes how these methods reveal clustering quality and the evolution of the active learning process across batches.
- **Supplementary Note 4: Details of Supervised Data Mining**
  - Offers an in-depth explanation of black-box interpretation tools (SHAP, PDP, Friedman’s H-statistics) used to understand model predictions.
  - Contains comprehensive visualizations and discussions that detail how feature contributions were assessed.
- **Supplementary Note 5: Details of Material Characterization**
  - Provides detailed protocols and instrument parameters for techniques such as BET, XRD, TEM, XPS, ICP, and EPR.
  - Explains how these characterization results support the observed structural and compositional modifications in the catalysts.
- **Supplementary Note 6: Details of Domain Adaptation-Assisted DFT Theoretical Simulation**
  - Documents the step-by-step process for generating the DFT dataset, training ML surrogate models, and conducting GA searches for stable doping configurations.
  - Includes specifics of the computational settings and the methods used for both implicit and explicit solvent modeling.
- **Supplementary Note 7: Details of Cross Domain Evaluation**
  - Provides an in-depth analysis of the performance and transferability of the ML models after domain adaptation, including potential issues like catastrophic forgetting.
- **Supplementary Note 8: Details of Broad Candidate Space Exploration**
  - Extends the discussion on dopant configurations and reveals additional insights from genetic algorithm searches, particularly highlighting effective dopant elements (e.g., Zr, Nb, Ta).

**Usage:**

- As stated in the manuscript, they offer optional, in-depth methodological details and extended data analysis for specialists interested in the technical nuances of the study.

**Machine Learning Databases and Script**

**Overview:**  
This repository contains the code we utilized. Each of these subdirectories corresponds to a distinct component of the research workflow – from active learning and DFT modeling to data mining and dimensionality reduction. The detailed description for each follow.

**1\. Adaptive_Active Learning Loop&GA Prediction**

This folder contains the Python Notebook scripts that manage the active learning (AL) loop and the accompanying genetic algorithm (GA) predictions. These scripts are used to iteratively update ML models and suggest new experimental formulations based on the highest prediction uncertainties and the lowest predicted overpotentials.

- **data_preprocessing_full.ipynb**  
    – Prepares and cleans the full experimental dataset for active learning.  
    – Contains steps for importing raw data, handling missing values, and formatting the data for model training.
- **data_preprocessing_high_quality.ipynb**  
    – Similar to the above but processes the high‐quality subset of the dataset.  
    – Ensures that only the data meeting rigorous quality criteria are included.
- **Model Retraining and GA Prediction of Highest Variance and Lowest Overpotential (Weighted Committee Ensemble).ipynb**  
    – Implements the retraining of the ML committee models using the latest experimental feedback in the AL loop  
    – Runs the genetic algorithm (GA) to search the parameter space for formulations that either maximize the uncertainty (variance) or minimize the predicted overpotential.  
    – Uses a weighted ensemble approach to combine predictions from multiple models.

**2\. AL_failure_tsne**

This folder focuses on documenting and visualizing the failures (i.e., experiments with poor performance) during the active learning iterations. The t-SNE (t-distributed stochastic neighbor embedding) technique is applied to reduce the dimensionality of encoded experimental parameters and visualize clusters of failure versus success.

- **1st_final.csv, 2nd_final.csv, … 5th_final.csv**  
    – CSV files containing the raw experimental results (over several AL iterations/batches) for each batch.
- **1st_final_encoded.csv, 2nd_final_encoded.csv, … 5th_final_encoded.csv**  
    – These files contain the encoded version of the corresponding batch data; that is, the high-dimensional experimental parameters transformed into a numerical representation suitable for t-SNE analysis.
- **unsupervised_tsne_plot_visulization.py**  
    – A Python script that reads the encoded CSV files and generates t-SNE plots.  
    – Helps users visualize how experimental samples are distributed in a reduced 2D space, identifying clusters corresponding to failure regions and guiding model improvements.

**3\. DFT Domain Adaptation**

This folder is dedicated to the domain adaptation work for density functional theory (DFT) simulations. The goal here is to adapt ML models trained on one domain (e.g., common elements) to predict properties for a broader or different domain (e.g., rare elements), thus reducing the cost of DFT calculations.

**Subfolders in DFT Domain Adaptation**

**a. A (Source Domain) and B (Target Domain) Descriptor Generation**

- **Generate the descriptors field A.ipynb**  
    – Generates numerical descriptors (features) for the source domain set set of DFT calculations, typically corresponding to one subset of the elemental candidates.
- **Generate the descriptors field B.ipynb**  
    – Similarly, this notebook generates descriptors for target domain representing another domain of elements.
- These descriptors serve as input features for ML surrogate models that predict slab energies.

**b. Cross Domain Evaluation**

- **Cross Domain Evaluation.ipynb**  
    – Evaluates the performance of models when applied across domains.  
    – Checks how well a model trained on one descriptor set (or domain) predicts outcomes in the other domain.

**c. Domain Adaptation (Transfer Learning)**

- **domain adaptation O random init 1~10.ipynb**
- **domain adaptation OH random init 1~10.ipynb**
- **domain adaptation OOH random init 1~10.ipynb**
- **domain adaptation SLAB random init 1~10.ipynb**  
    – Each notebook here applies transfer learning (domain adaptation) for different species or configurations:
  - “O”, “OH”, and “OOH” refer to the slab with different oxygen-related intermediates adsorbed.
  - “SLAB” refers to the energy of the catalyst slab. – “1~10” are internal serial number given to different ML architectures in the committee indicates that multiple random initializations are used to ensure robustness.

**d. GA search**

- **GA_search_A~D (C example).ipynb**  
    – Runs a GA search to identify the optimum doping configuration for sample “C” (as an example).  
    – This notebook demonstrates how the GA is applied to search for the lowest predicted slab energy.
- **GA_search_broad_random.ipynb**  
    – Production scripts that we conducted a broader GA search over a larger parameter space like population and generation to get the results for further DFT simulation of surface reaction pathways.

**e. Model Committee Training**

This subfolder contains notebooks for training ensemble (committee) models on two separate datasets (A and B, namely source and target domain).

- **Committee A on dataset A**  
    – Notebooks include:
  - **sm,ew,acsf 3 rcut 6 O 1~10 A.ipynb**  
        – Trains models on source domain for predicting energy of O species adsorbed slab.
  - **sm,ew,acsf 3 rcut 6 OH 1~10 A.ipynb**  
        – For hydroxyl (OH) species.
  - **sm,ew,acsf 3 rcut 6 OOH 1~10 A.ipynb**  
        – For OOH species.
  - **sm,ew,acsf 3 rcut 6 SLAB 1~10 A.ipynb**  
        – For pure clean slab energy. – “sm, ew, acsf” likely denote the types of descriptors used (in ACSF); “rcut 6” indicates a cutoff radius; “1~10” are numbers for different architectures.
- **Committee B on dataset B**  
    – Contains similar notebooks as for Committee A, but for the target domain itself:
  - **sm,ew,acsf 3 rcut 6 O 1~10 B.ipynb**
  - **sm,ew,acsf 3 rcut 6 OH 1~10 B.ipynb**
  - **sm,ew,acsf 3 rcut 6 OOH 1~10 B.ipynb**
  - **sm,ew,acsf 3 rcut 6 SLAB 1~10 B.ipynb**

**f. statistic whole element search**

**this part corresponds to supplementary**

- **RuO2_110_std.cif**  
    – A crystallographic file containing the standard structure for RuO₂ (specifically the 110 surface) used as a reference.
- **Statistic of broad candidate space search.ipynb**  
    – A notebook that performs statistical analysis over the broad candidate space for multi-metal formulations.
- **Whole Element 50percent search.csv**  
    – A CSV file summarizing search results when exploring candidate formulations with a 50% threshold in element proportions.

**4\. DFT Structures Generate**

This folder stores scripts that generate the DFT structure files used in subsequent simulations.

- **Element Set A_Data Set A_Generate SLAB.py**  
    – A Python script that constructs the DFT slab structures for “Element Set A” (Dataset A).
- **Element Set B_Data Set B_Generate SLAB.py**  
    – Similar script for “Element Set B” (Dataset B).
- **Generate Oxygen Intermediate_O_OH_OOH.py**  
    – A script to generate structures of oxygen-related intermediates (O, OH, and OOH) on the catalyst surface.  
    – These are used as input geometries in subsequent DFT calculations.

**5\. Domain Knowledge Based Initial ML Committee and Blackbox Interpretation**

This folder contains the scripts and notebooks that build the initial machine learning committees from the literature‐derived (domain knowledge) datasets and perform “blackbox” interpretation of their predictions using methods like SHAP.

**Subfolders:**

**a. Classification (LSE)**

This branch is for classification tasks related to level set estimation (LSE) on the domain knowledge dataset.

- **Activity**  
    – Contains three subfolders:
  - **200**: Contains files (notebooks, scripts, and possibly output files) specific to classification when the overpotential threshold is 200 mV.
  - **250**: Similar content for the 250 mV threshold.
  - **300**: For the 300 mV threshold. – In each folder, you will find notebooks dedicated to initial ML committee training and a “blackbox interpretation” subfolder (which explains model decisions using interpretation tools).
- **Stability**  
    – Contains three subfolders for stability classification corresponding to different decay rate labels:
  - **+1**: For a positive classification (e.g., high decay rate).
  - **\-1**: For a negative classification.
  - **0**: For a neutral or baseline stability category. – Each contains similar structure (blackbox interpretation and initial ML committee training).

**b. Regression**

This branch focuses on regression tasks using the domain knowledge dataset.

- **Activity**  
    – Contains notebooks and scripts that perform regression (predicting overpotentials) along with blackbox interpretation (e.g., SHAP, PDP) to understand feature importance.
- **Stability**  
    – Contains regression scripts for predicting stability (voltage decay, etc.) and the corresponding interpretation files.

**6\. Domain Knowledge Database Preprocessing**

This folder holds the preprocessing steps and processed data files generated from the literature‐derived domain knowledge dataset.

**Subfolders:**

**a. Activity**

- **Correlation Matrix.ipynb**  
    – A notebook that computes and visualizes the correlation matrix for the activity dataset.
- **database_full_ac.pkl** and **database_high_quality_ac.pkl**  
    – Serialized (pickle) files containing the full and high-quality versions of the activity dataset, respectively.
- **data_preprocessing_full.ipynb** and **data_preprocessing_high_quality.ipynb**  
    – Notebooks that document the preprocessing pipeline (cleaning, normalization, formatting) for the full and high-quality activity datasets.
- **OER_activity.csv**  
    – A CSV file with the final, processed activity data (overpotential, synthesis parameters, etc.).

**b. Stability**

- **Correlation Matrix.ipynb**  
    – Similar to the activity folder, this notebook visualizes correlations among variables in the stability dataset.
- **database_full_st.pkl** and **database_high_quality_st.pkl**  
    – Pickle files for the full and high-quality stability datasets.
- **data_preprocessing_full.ipynb** and **data_preprocessing_high_quality.ipynb**  
    – Notebooks for cleaning and processing the stability data.
- **OER_stability.csv**  
    – CSV file containing the processed stability data.

**7\. Half-Cell Stability Quick Predict**

This folder provides quick-prediction tools for the half-cell stability tests using the ML models.

- **400.csv** and **500.csv**  
    – CSV files containing experimental data for half-cell tests at annealing temperatures of 400 °C and 500 °C, respectively.
- **database_high_quality_st.pkl**  
    – A high-quality stability dataset (pickled) used for predictions.
- **High quality predict stability.ipynb**  
    – A notebook that applies the ML models to predict long-term stability performance from the high-quality stability dataset.

**8\. Number of Possibilities Calculation**

This folder contains the scripts used to calculate the combinatorial space of possible catalyst formulations.

- **Possibility Number Calculation.ipynb**  
    – A notebook that implements the mathematical calculation of the number of combinations available when selecting four metals out of 58 and assigning integer percentage values (with the sum equal to 100%), along with other synthesis parameters.  
    – This script shows that the candidate space is enormous (in the billions to trillions), justifying the use of active learning and ML techniques.

**9\. PCA and t-SNE Dimensional Reduction Representation**

This folder contains notebooks and data files used for dimensionality reduction analysis.

- **database_full_ac.pkl, database_full_st.pkl, database_high_quality_ac.pkl, database_high_quality_st.pkl**  
    – These pickle files are reused from the Domain Knowledge Database Preprocessing folder.
- **PCA&TSNE.ipynb**  
    – A notebook that applies Principal Component Analysis (PCA) and t-SNE to the activity and stability datasets. (to be noted, different from AL failure mode analysis t-sne)  
    – It generates reduced-dimension representations for visualization and further data exploration.

**10\. Unsupervised Data Mining**

This folder contains scripts and outputs for unsupervised data mining methods applied to the domain knowledge datasets. It is divided into two main parts: Apriori association rule mining and bibliometric network graph generation.

**Subfolders:**

**a. Apriori associate rule mining**

This section performs association rule mining to discover relationships among synthesis parameters and elemental compositions.

- **Activity**  
    – **Apriori_High_Full.ipynb** and **Apriori_High_Quality.ipynb**  
    • Notebooks that run the Apriori algorithm on the full and high-quality activity datasets, respectively. – CSV files:  
    • **OER_activity.csv** (the processed activity dataset used for mining, namely from publications domain knowledge in tabular form)  
    • Multiple CSVs such as **three_set_200_Full.csv**, **three_set_200_High_Quality.csv**, **three_set_250_Full.csv**, **three_set_250_High_Quality.csv**, **three_set_300_Full.csv**, **three_set_300_High_Quality.csv**, and similarly for two-set combinations (e.g., **two_set_200_Full.csv**, etc.).  
    • These files report the frequency and lift values for combinations (itemsets) of elements and parameters at different overpotential thresholds (200, 250, 300 mV).
- **Stability**  
    – **Apriori_Full.ipynb** and **Apriori_High_Quality.ipynb**  
    • Notebooks for running the association rule mining on stability data. – CSV files:  
    • **OER_stability.csv** (processed stability dataset)  
    • CSVs with names such as **three_set_0_Full.csv**, **three_set_0_High_Quality.csv**, **three_set_m1_Full.csv**, **three_set_m1_High_Quality.csv**, **three_set_p1_Full.csv**, **three_set_p1_High_Quality.csv**, and the corresponding “two_set” CSV files.  
    • These detail the association rules for stability metrics under different decay rate categories.

**b. Bibliometric Interconnected Network Graph**

This section creates network graphs to visualize co-occurrence and relationships among the elemental and synthesis parameters extracted from the literature.

- **Activity**  
    – **activity_full.ipynb** and **activity_high_quality.ipynb**  
    • Notebooks that generate network graphs for the full and high-quality activity datasets. – CSV files:  
    • **Coexistence.csv** and **Coexistence_HQ.csv** – Contain data on co-occurrence frequencies of elements.  
    • **Occurence_Element_Total.csv** and **Occurence_Element_Total_HQ.csv** – Summarize the total occurrence counts for each element in the datasets.  
    • **OER_activity.csv** – Provided again as the underlying data.
- **Stability**  
    – Contains similar files:  
    • CSV files for coexistence and occurrence (e.g., **Coexistence.csv**, **Coexistence_HQ.csv**, **Occurence_Element_Total.csv**, **Occurence_Element_Total_HQ.csv**, and **OER_stability.csv**).  
    • **stability_full.ipynb** and **stability_high_quality.ipynb**  
    – Notebooks that create the network graphs for stability data.
