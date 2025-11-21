import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import sqlite3
from datetime import datetime
import json
import io
import base64

# Analysis libraries
try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    from sklearn.impute import KNNImputer, SimpleImputer
    import matplotlib.pyplot as plt
    import seaborn as sns
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Touchless Survey System",
    page_icon="✋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default admin password (NOT shown to users)
DEFAULT_ADMIN_PASSWORD = "admin123"

# Database file
DB_FILE = "survey_responses.db"

# Survey questions
SURVEY_QUESTIONS = [
    "How satisfied are you with the workshop content?",
    "How satisfied are you with the instructor's teaching?",
    "How satisfied are you with the workshop materials?",
    "How satisfied are you with the hands-on activities?",
    "How satisfied are you with the overall workshop experience?"
]

# Gesture mapping
GESTURE_MAP = {
    'thumbs_up': {'label': 'Satisfied', 'score': 4, 'emoji': '👍'},
    'heart_sign': {'label': 'Very Satisfied', 'score': 5, 'emoji': '❤️'},
    'thumbs_down': {'label': 'Unsatisfied', 'score': 2, 'emoji': '👎'},
    'waving_finger': {'label': 'Very Unsatisfied', 'score': 1, 'emoji': '☝️'},
    'closed_fist': {'label': 'No Answer', 'score': None, 'emoji': '✊'}
}

# ============================================================================
# EDUCATIONAL CONTENT - IMPUTATION STRATEGIES
# ============================================================================

IMPUTATION_STRATEGIES = {
    'mean': {
        'name': 'Mean Imputation',
        'description': 'Replaces missing values with the mean (average) of the column.',
        'when_to_use': 'Best for normally distributed data without outliers.',
        'pros': 'Simple, fast, preserves the mean of the dataset.',
        'cons': 'Reduces variance, ignores relationships between variables.',
        'example': 'If scores are [1, 2, ?, 4, 5], missing value becomes 3.0'
    },
    'median': {
        'name': 'Median Imputation',
        'description': 'Replaces missing values with the median (middle value) of the column.',
        'when_to_use': 'Best when data has outliers or is skewed.',
        'pros': 'Robust to outliers, preserves central tendency.',
        'cons': 'Reduces variance, ignores relationships between variables.',
        'example': 'If scores are [1, 2, ?, 4, 100], missing value becomes 3.0 (not affected by 100)'
    },
    'mode': {
        'name': 'Mode Imputation',
        'description': 'Replaces missing values with the most frequent value.',
        'when_to_use': 'Best for categorical data or discrete scores.',
        'pros': 'Preserves the most common response pattern.',
        'cons': 'Can overrepresent the mode, not suitable for continuous data.',
        'example': 'If scores are [5, 4, ?, 5, 5, 3], missing value becomes 5'
    },
    'forward_fill': {
        'name': 'Forward Fill (FFill)',
        'description': 'Fills missing values with the previous valid observation.',
        'when_to_use': 'Best for time-series data or sequential responses.',
        'pros': 'Maintains continuity, no calculation needed.',
        'cons': 'Assumes pattern continues, not suitable for random missing data.',
        'example': 'If sequence is [3, 4, ?, ?, 5], missing values become [3, 4, 4, 4, 5]'
    },
    'backward_fill': {
        'name': 'Backward Fill (BFill)',
        'description': 'Fills missing values with the next valid observation.',
        'when_to_use': 'Best for time-series when future value is more relevant.',
        'pros': 'Uses future information, maintains continuity.',
        'cons': 'Assumes reverse pattern, not suitable for random missing data.',
        'example': 'If sequence is [3, ?, ?, 5, 4], missing values become [3, 5, 5, 5, 4]'
    },
    'interpolate': {
        'name': 'Linear Interpolation',
        'description': 'Estimates missing values by drawing a line between known values.',
        'when_to_use': 'Best for continuous data with smooth trends.',
        'pros': 'Creates smooth transitions, uses surrounding context.',
        'cons': 'Assumes linear relationship, needs values on both sides.',
        'example': 'If scores are [2, ?, ?, 6], missing values become [2, 3.33, 4.67, 6]'
    },
    'zero': {
        'name': 'Fill with Zero',
        'description': 'Replaces all missing values with 0.',
        'when_to_use': 'When missing means "no response" or "zero activity".',
        'pros': 'Simple, explicit meaning.',
        'cons': 'Can distort statistics, may not make sense for ratings.',
        'example': 'Missing values in satisfaction ratings might not mean "0 satisfaction"'
    },
    'constant': {
        'name': 'Fill with Constant',
        'description': 'Replaces missing values with a specific constant value (e.g., 3 for neutral).',
        'when_to_use': 'When you want to assign a specific meaning to missing data.',
        'pros': 'Explicit control, can represent "neutral" or "no opinion".',
        'cons': 'Arbitrary choice can bias results.',
        'example': 'Fill missing satisfaction with 3 (neutral on 1-5 scale)'
    },
    'knn': {
        'name': 'KNN Imputation',
        'description': 'Uses K-Nearest Neighbors to estimate missing values based on similar responses.',
        'when_to_use': 'When variables are related and you have enough complete cases.',
        'pros': 'Considers relationships, more sophisticated.',
        'cons': 'Computationally expensive, requires complete cases.',
        'example': 'If respondents with similar Q1-Q3 scores tend to give similar Q4 scores, use those patterns'
    },
    'group_mean': {
        'name': 'Group Mean Imputation',
        'description': 'Fills missing values with the mean of a specific group (e.g., by organization).',
        'when_to_use': 'When groups have different response patterns.',
        'pros': 'Preserves group differences, more contextual.',
        'cons': 'Requires meaningful grouping variable.',
        'example': 'Use average score from same organization instead of overall average'
    }
}

# ============================================================================
# EDUCATIONAL CONTENT - STATISTICAL METHODS
# ============================================================================

STATISTICAL_METHODS = {
    'descriptive': {
        'name': 'Descriptive Statistics',
        'description': 'Basic summary statistics including mean, median, standard deviation, min, max.',
        'purpose': 'Understand the central tendency and spread of your data.',
        'what_you_learn': 'Average satisfaction levels, consistency of responses, range of ratings.',
        'interpretation': 'High mean = good satisfaction. Low std dev = consistent responses.',
    },
    'normality': {
        'name': 'Normality Test (Shapiro-Wilk)',
        'description': 'Tests if your data follows a normal (bell curve) distribution.',
        'purpose': 'Determine if you can use parametric statistical tests.',
        'what_you_learn': 'Whether scores are normally distributed (most scores near middle vs scattered).',
        'interpretation': 'p > 0.05: Data is normal. p < 0.05: Data is not normal.',
    },
    'correlation': {
        'name': 'Correlation Analysis',
        'description': 'Measures relationships between different questions.',
        'purpose': 'Understand which aspects of the workshop are related.',
        'what_you_learn': 'Do people who rate content high also rate instruction high?',
        'interpretation': 'Values close to 1: Strong positive relationship. Close to -1: Inverse relationship. Close to 0: No relationship.',
    },
    'ttest': {
        'name': 'T-Test',
        'description': 'Compares means between two groups to see if they are significantly different.',
        'purpose': 'Test if satisfaction differs between groups (e.g., organizations).',
        'what_you_learn': 'Is the difference in satisfaction real or just by chance?',
        'interpretation': 'p < 0.05: Groups are significantly different.',
    },
    'anova': {
        'name': 'ANOVA (Analysis of Variance)',
        'description': 'Compares means across multiple groups (3 or more).',
        'purpose': 'Test if satisfaction differs across multiple organizations/groups.',
        'what_you_learn': 'Which groups have significantly different satisfaction levels?',
        'interpretation': 'p < 0.05: At least one group is significantly different.',
    },
    'chi_square': {
        'name': 'Chi-Square Test',
        'description': 'Tests relationships between categorical variables.',
        'purpose': 'See if satisfaction categories relate to organization or other factors.',
        'what_you_learn': 'Are certain organizations more likely to give high ratings?',
        'interpretation': 'p < 0.05: Significant relationship exists.',
    }
}

# ============================================================================
# EDUCATIONAL CONTENT - MACHINE LEARNING MODELS
# ============================================================================

ML_MODELS = {
    'logistic': {
        'name': 'Logistic Regression',
        'description': 'Predicts binary outcomes (satisfied vs unsatisfied) using a linear approach.',
        'purpose': 'Understand which factors predict satisfaction and their importance.',
        'when_to_use': 'Best for binary classification with interpretable results.',
        'strengths': 'Fast, interpretable, shows feature importance, works well with small datasets.',
        'limitations': 'Assumes linear relationships, limited to binary/simple outcomes.',
        'what_you_learn': 'Which questions are most predictive of overall satisfaction?'
    },
    'decision_tree': {
        'name': 'Decision Tree',
        'description': 'Creates a tree of decisions to classify satisfaction levels.',
        'purpose': 'Visual understanding of decision rules for satisfaction.',
        'when_to_use': 'When you want easily interpretable rules.',
        'strengths': 'Easy to visualize and explain, handles non-linear patterns.',
        'limitations': 'Can overfit, unstable with small changes in data.',
        'what_you_learn': 'Clear if-then rules: "If Q1 < 3 and Q2 < 4, then unsatisfied"'
    },
    'random_forest': {
        'name': 'Random Forest',
        'description': 'Combines multiple decision trees for more robust predictions.',
        'purpose': 'Get more accurate predictions by averaging many decision trees.',
        'when_to_use': 'When you want high accuracy and feature importance.',
        'strengths': 'Very accurate, robust, provides feature importance, handles outliers well.',
        'limitations': 'Less interpretable than single tree, slower to train.',
        'what_you_learn': 'Most important factors for satisfaction with high accuracy.'
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'description': 'Builds trees sequentially, each correcting errors of previous ones.',
        'purpose': 'Achieve highest possible prediction accuracy.',
        'when_to_use': 'When accuracy is most important and you have enough data.',
        'strengths': 'Often best performance, captures complex patterns.',
        'limitations': 'Can overfit, requires careful tuning, slower training.',
        'what_you_learn': 'Complex patterns in satisfaction with very high accuracy.'
    },
    'svm': {
        'name': 'Support Vector Machine (SVM)',
        'description': 'Finds the best boundary between satisfied and unsatisfied responses.',
        'purpose': 'Classify with maximum separation between classes.',
        'when_to_use': 'When classes are well-separated and you have clean data.',
        'strengths': 'Effective in high dimensions, memory efficient.',
        'limitations': 'Slow on large datasets, less interpretable.',
        'what_you_learn': 'Clear separation boundary between satisfied and unsatisfied.'
    },
    'knn': {
        'name': 'K-Nearest Neighbors (KNN)',
        'description': 'Predicts satisfaction based on most similar responses.',
        'purpose': 'Use similarity to past responses for prediction.',
        'when_to_use': 'When similar response patterns should give similar outcomes.',
        'strengths': 'Simple concept, no training time, works with irregular patterns.',
        'limitations': 'Slow prediction, sensitive to scale, needs good K value.',
        'what_you_learn': '"People with similar responses tend to have similar satisfaction"'
    },
    'naive_bayes': {
        'name': 'Naive Bayes',
        'description': 'Uses probability theory to predict satisfaction likelihood.',
        'purpose': 'Fast probabilistic classification.',
        'when_to_use': 'When you want quick results and probability estimates.',
        'strengths': 'Very fast, works well with small data, provides probabilities.',
        'limitations': 'Assumes independence (naive assumption), less accurate.',
        'what_you_learn': 'Probability of satisfaction given certain responses.'
    }
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create responses table
    c.execute('''CREATE TABLE IF NOT EXISTS responses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  name TEXT,
                  organization TEXT,
                  q1_label TEXT, q1_score REAL, q1_confidence REAL,
                  q2_label TEXT, q2_score REAL, q2_confidence REAL,
                  q3_label TEXT, q3_score REAL, q3_confidence REAL,
                  q4_label TEXT, q4_score REAL, q4_confidence REAL,
                  q5_label TEXT, q5_score REAL, q5_confidence REAL,
                  overall_score REAL)''')
    
    # Create settings table
    c.execute('''CREATE TABLE IF NOT EXISTS settings
                 (key TEXT PRIMARY KEY,
                  value TEXT)''')
    
    # Create interpretations table
    c.execute('''CREATE TABLE IF NOT EXISTS interpretations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  analysis_type TEXT,
                  interpretation TEXT,
                  timestamp TEXT)''')
    
    conn.commit()
    conn.close()

def save_response(name, org, responses):
    """Save survey response to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Calculate overall score
    scores = [r['score'] for r in responses if r['score'] is not None]
    overall_score = sum(scores) / len(scores) if scores else None
    
    # Prepare data - 19 values total
    data = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        name,
        org,
    ]
    
    for r in responses:
        data.extend([r['label'], r['score'], r['confidence']])
    
    data.append(overall_score)
    
    c.execute('''INSERT INTO responses (timestamp, name, organization, 
                  q1_label, q1_score, q1_confidence,
                  q2_label, q2_score, q2_confidence,
                  q3_label, q3_score, q3_confidence,
                  q4_label, q4_score, q4_confidence,
                  q5_label, q5_score, q5_confidence,
                  overall_score) 
                  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
    conn.commit()
    conn.close()

def get_all_responses():
    """Get all responses from database"""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM responses", conn)
    conn.close()
    return df

def get_setting(key, default=None):
    """Get setting from database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else default

def save_setting(key, value):
    """Save setting to database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def save_interpretation(analysis_type, interpretation):
    """Save interpretation note"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute("INSERT INTO interpretations (analysis_type, interpretation, timestamp) VALUES (?, ?, ?)",
              (analysis_type, interpretation, timestamp))
    conn.commit()
    conn.close()

def get_interpretation(analysis_type):
    """Get latest interpretation for analysis type"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT interpretation FROM interpretations WHERE analysis_type=? ORDER BY timestamp DESC LIMIT 1",
              (analysis_type,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else ""

def delete_response(response_id):
    """Delete a response"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM responses WHERE id=?", (response_id,))
    conn.commit()
    conn.close()

def clear_all_responses():
    """Clear all responses"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM responses")
    conn.commit()
    conn.close()

def generate_synthetic_data(num_responses, satisfied_pct, diversity):
    """Generate synthetic survey data for testing"""
    import random
    
    # Names pool
    first_names = ['Alex', 'Jordan', 'Taylor', 'Morgan', 'Casey', 'Riley', 'Avery', 'Quinn',
                   'Jamie', 'Charlie', 'Sam', 'Drew', 'Blake', 'Sage', 'River', 'Dakota',
                   'Phoenix', 'Skylar', 'Jesse', 'Kai', 'Rowan', 'Finley', 'Emerson', 'Parker']
    
    organizations = ['PSITE', 'ACM', 'IEEE', 'Tech Corp', 'Data Labs', 'AI Institute', 
                    'Cloud Systems', 'Analytics Hub', 'Innovation Center', 'Research Group']
    
    gestures = {
        5: {'label': 'Very Satisfied', 'emoji': '❤️'},
        4: {'label': 'Satisfied', 'emoji': '👍'},
        2: {'label': 'Unsatisfied', 'emoji': '👎'},
        1: {'label': 'Very Unsatisfied', 'emoji': '☝️'}
    }
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    for i in range(num_responses):
        # Determine if this response should be satisfied
        is_satisfied = random.random() < (satisfied_pct / 100)
        
        # Generate scores based on satisfaction and diversity
        if diversity == 'Low':
            # Low diversity - scores very similar
            if is_satisfied:
                base_score = random.choice([4, 5])
                scores = [base_score + random.choice([0, 0, 0, 1, -1]) for _ in range(5)]
            else:
                base_score = random.choice([1, 2])
                scores = [base_score + random.choice([0, 0, 0, 1, -1]) for _ in range(5)]
        elif diversity == 'Medium':
            # Medium diversity - some variation
            if is_satisfied:
                scores = [random.choice([3, 4, 4, 4, 5, 5]) for _ in range(5)]
            else:
                scores = [random.choice([1, 1, 2, 2, 2, 3]) for _ in range(5)]
        else:  # High diversity
            # High diversity - wide range
            if is_satisfied:
                scores = [random.randint(3, 5) for _ in range(5)]
            else:
                scores = [random.randint(1, 3) for _ in range(5)]
        
        # Clip scores to valid range
        scores = [max(1, min(5, s)) for s in scores]
        
        # Occasionally add missing values
        if random.random() < 0.1:  # 10% chance
            scores[random.randint(0, 4)] = None
        
        # Generate response data
        name = random.choice(first_names) + str(random.randint(1, 99))
        org = random.choice(organizations)
        overall_score = sum(s for s in scores if s is not None) / len([s for s in scores if s is not None])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        data = [timestamp, name, org]
        
        for score in scores:
            if score is None:
                data.extend(['No Answer', None, random.uniform(0.7, 0.9)])
            else:
                gesture_info = gestures[score]
                data.extend([gesture_info['label'], score, random.uniform(0.85, 0.99)])
        
        data.append(overall_score)
        
        c.execute('''INSERT INTO responses (timestamp, name, organization, 
                      q1_label, q1_score, q1_confidence,
                      q2_label, q2_score, q2_confidence,
                      q3_label, q3_score, q3_confidence,
                      q4_label, q4_score, q4_confidence,
                      q5_label, q5_score, q5_confidence,
                      overall_score) 
                      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
    
    conn.commit()
    conn.close()

def update_responses_with_cleaned(df_clean):
    """Update database with cleaned data"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    for idx, row in df_clean.iterrows():
        # Recalculate overall score
        scores = [row[col] for col in score_cols if pd.notna(row[col])]
        overall_score = sum(scores) / len(scores) if scores else None
        
        # Update the record
        c.execute('''UPDATE responses 
                     SET q1_score=?, q2_score=?, q3_score=?, q4_score=?, q5_score=?, overall_score=?
                     WHERE id=?''',
                  (row['q1_score'], row['q2_score'], row['q3_score'], 
                   row['q4_score'], row['q5_score'], overall_score, row['id']))
    
    conn.commit()
    conn.close()

# ============================================================================
# ENHANCED CLEANING FUNCTIONS
# ============================================================================

def apply_imputation(df, strategy='median', constant_value=3, group_col=None):
    """Apply selected imputation strategy"""
    df_clean = df.copy()
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    # Ensure numeric
    for col in score_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    if strategy == 'mean':
        for col in score_cols:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    elif strategy == 'median':
        for col in score_cols:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    elif strategy == 'mode':
        for col in score_cols:
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
    
    elif strategy == 'forward_fill':
        df_clean[score_cols] = df_clean[score_cols].fillna(method='ffill')
    
    elif strategy == 'backward_fill':
        df_clean[score_cols] = df_clean[score_cols].fillna(method='bfill')
    
    elif strategy == 'interpolate':
        for col in score_cols:
            df_clean[col] = df_clean[col].interpolate(method='linear')
    
    elif strategy == 'zero':
        df_clean[score_cols] = df_clean[score_cols].fillna(0)
    
    elif strategy == 'constant':
        df_clean[score_cols] = df_clean[score_cols].fillna(constant_value)
    
    elif strategy == 'knn':
        if ML_AVAILABLE:
            imputer = KNNImputer(n_neighbors=3)
            df_clean[score_cols] = imputer.fit_transform(df_clean[score_cols])
    
    elif strategy == 'group_mean' and group_col and group_col in df_clean.columns:
        for col in score_cols:
            df_clean[col] = df_clean.groupby(group_col)[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    # Fill any remaining NaN with median as fallback
    for col in score_cols:
        if df_clean[col].isnull().any():
            median = df_clean[col].median()
            df_clean[col].fillna(median if pd.notna(median) else 3.0, inplace=True)
    
    return df_clean

# ============================================================================
# ENHANCED ANALYSIS FUNCTIONS
# ============================================================================

def perform_statistical_analysis(df, methods=['descriptive']):
    """Perform selected statistical analyses"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    results = {}
    
    if 'descriptive' in methods:
        results['descriptive'] = {
            'mean': df[score_cols].mean().mean(),
            'std': df[score_cols].std().mean(),
            'median': df[score_cols].median().median(),
            'per_question': {}
        }
        for i, col in enumerate(score_cols, 1):
            results['descriptive']['per_question'][f'Q{i}'] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max()
            }
    
    if 'normality' in methods and len(df) >= 3:
        results['normality'] = {}
        for i, col in enumerate(score_cols, 1):
            if df[col].std() > 0:
                try:
                    stat, p = stats.shapiro(df[col])
                    results['normality'][f'Q{i}'] = {
                        'statistic': stat,
                        'p_value': p,
                        'is_normal': p > 0.05,
                        'interpretation': f"Data is {'normal' if p > 0.05 else 'not normal'} (p={p:.4f})"
                    }
                except:
                    results['normality'][f'Q{i}'] = {'error': 'Cannot compute'}
    
    if 'correlation' in methods and len(df) >= 3:
        results['correlation'] = df[score_cols].corr()
    
    return results

def train_ml_model(df, model_type='logistic'):
    """Train selected ML model"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    X = df[score_cols].values
    y = (df['overall_score'] >= 4).astype(int).values
    
    # Check class distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) < 2:
        satisfied_count = (y == 1).sum()
        unsatisfied_count = (y == 0).sum()
        return {
            'error': f'Need both satisfied AND unsatisfied responses for ML training.\n\n'
                    f'Current data:\n'
                    f'  • Satisfied (score ≥ 4): {satisfied_count} responses\n'
                    f'  • Unsatisfied (score < 4): {unsatisfied_count} responses\n\n'
                    f'💡 Tip: To use ML, you need at least 1 response in each category.\n'
                    f'Try collecting more diverse feedback!'
        }
    
    # Adjust test_size if we have very few samples
    test_size = 0.3 if len(df) >= 10 else 0.2
    if len(df) < 5:
        test_size = 1 / len(df)  # Leave one out
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Select model
    models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=4),
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'svm': SVC(random_state=42, probability=True),
        'knn': KNeighborsClassifier(n_neighbors=min(3, len(df)-1)),
        'naive_bayes': GaussianNB()
    }
    
    model = models.get(model_type, LogisticRegression(random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results = {
        'model_type': model_type,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_'):
        results['feature_importance'] = dict(zip(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 
                                                  model.feature_importances_.tolist()))
    elif hasattr(model, 'coef_'):
        results['feature_importance'] = dict(zip(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], 
                                                  model.coef_[0].tolist()))
    
    return results

def plot_basic_stats(df):
    """Create basic statistics plots"""
    score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Survey Statistics Dashboard', fontsize=14, fontweight='bold')
    
    # 1. Average scores
    means = [df[col].mean() for col in score_cols]
    colors = ['#2ecc71' if m>=4 else '#f39c12' if m>=3 else '#e74c3c' for m in means]
    axes[0,0].bar(range(5), means, color=colors, edgecolor='black', alpha=0.7)
    axes[0,0].axhline(4, color='green', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Average Scores per Question')
    axes[0,0].set_xticks(range(5))
    axes[0,0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    axes[0,0].set_ylim(0, 5.5)
    
    # 2. Distribution
    all_scores = df[score_cols].values.flatten()
    axes[0,1].hist(all_scores, bins=5, edgecolor='black', color='#3498db', alpha=0.7)
    axes[0,1].set_title('Score Distribution')
    axes[0,1].set_xlabel('Score')
    
    # 3. Box plot
    df[score_cols].boxplot(ax=axes[1,0])
    axes[1,0].set_title('Score Spread by Question')
    
    # 4. Satisfaction rate
    satisfied = (df['overall_score'] >= 4).sum()
    not_satisfied = (df['overall_score'] < 4).sum()
    axes[1,1].bar(['Not Satisfied', 'Satisfied'], [not_satisfied, satisfied],
                  color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Satisfaction Distribution')
    axes[1,1].set_ylabel('Count')
    
    plt.tight_layout()
    return fig

# ============================================================================
# SIMPLE PREDICTION (Replace with Teachable Machine)
# ============================================================================

def simple_predict(image):
    """Simple prediction - replace with Teachable Machine API call"""
    import random
    gestures = list(GESTURE_MAP.keys())
    gesture = random.choice(gestures)
    confidence = random.uniform(0.7, 0.99)
    return gesture, confidence

# ============================================================================
# ADMIN PANEL
# ============================================================================

def show_method_info(method_dict, method_key):
    """Display educational information about a method"""
    info = method_dict[method_key]
    with st.expander(f"📚 Learn about {info['name']}", expanded=False):
        st.markdown(f"**Description:** {info['description']}")
        st.markdown(f"**{list(info.keys())[3]}:** {list(info.values())[3]}")
        st.markdown(f"**{list(info.keys())[4]}:** {list(info.values())[4]}")
        if len(info) > 5:
            st.markdown(f"**{list(info.keys())[5]}:** {list(info.values())[5]}")
        if 'example' in info:
            st.info(f"💡 Example: {info['example']}")

def admin_panel():
    """Enhanced admin panel with educational features"""
    st.title("🔧 Admin Panel")
    
    # Check admin authentication - NO DEFAULT PASSWORD MESSAGE SHOWN
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.subheader("🔐 Admin Login")
        password = st.text_input("Enter Admin Password:", type="password", key="admin_pw")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Login", type="primary"):
                stored_password = get_setting('admin_password', DEFAULT_ADMIN_PASSWORD)
                if password == stored_password:
                    st.session_state.admin_authenticated = True
                    st.success("✓ Authenticated!")
                    st.rerun()
                else:
                    st.error("✗ Incorrect password")
        
        # NO default password hint shown to users
        return
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()
    
    # Tabs for different admin functions
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "⚙️ Settings", 
        "📊 View Data", 
        "🧹 Clean Data", 
        "📈 Statistics", 
        "🤖 Machine Learning"
    ])
    
    # ========== TAB 1: SETTINGS ==========
    with tab1:
        st.subheader("Application Settings")
        
        # Teachable Machine Model URL
        st.markdown("### 🎯 Teachable Machine Model")
        current_model_url = get_setting('model_url', '')
        model_url = st.text_input(
            "Teachable Machine Shareable Link:",
            value=current_model_url,
            help="Paste your Teachable Machine model share link here"
        )
        
        if st.button("💾 Save Model URL"):
            save_setting('model_url', model_url)
            st.success("✓ Model URL saved!")
        
        # Admin Password
        st.markdown("### 🔑 Change Admin Password")
        new_password = st.text_input("New Password:", type="password", key="new_pw")
        confirm_password = st.text_input("Confirm Password:", type="password", key="confirm_pw")
        
        if st.button("🔄 Update Password"):
            if new_password and new_password == confirm_password:
                save_setting('admin_password', new_password)
                st.success("✓ Password updated successfully!")
                st.info("⚠️ Remember your new password - there is no recovery option!")
            else:
                st.error("✗ Passwords don't match or are empty")
        
        # Survey Settings
        st.markdown("### 📋 Survey Settings")
        survey_title = st.text_input("Survey Title:", value=get_setting('survey_title', 'Touchless Satisfaction Survey'))
        if st.button("💾 Save Title"):
            save_setting('survey_title', survey_title)
            st.success("✓ Title saved!")
    
    # ========== TAB 2: VIEW DATA ==========
    with tab2:
        st.subheader("📊 Survey Responses")
        
        df = get_all_responses()
        
        if len(df) == 0:
            st.info("No responses yet. Start collecting survey data!")
        else:
            st.success(f"Total Responses: {len(df)}")
            
            # Display data
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                "survey_responses.csv",
                "text/csv",
                key='download-csv'
            )
            
            # Delete options
            st.markdown("### 🗑️ Data Management")
            col1, col2 = st.columns(2)
            
            with col1:
                response_id = st.number_input("Delete Response ID:", min_value=1, step=1)
                if st.button("🗑️ Delete Response"):
                    delete_response(response_id)
                    st.success(f"✓ Deleted response {response_id}")
                    st.rerun()
            
            with col2:
                if st.button("⚠️ Clear All Data", type="secondary"):
                    if st.checkbox("Confirm deletion"):
                        clear_all_responses()
                        st.success("✓ All data cleared")
                        st.rerun()
        
        # Add upload and synthetic data options
        st.markdown("---")
        st.markdown("### 🧪 Testing & Data Import")
        
        tab_upload, tab_synthetic = st.tabs(["📤 Upload CSV", "🎲 Generate Synthetic Data"])
        
        with tab_upload:
            st.markdown("#### Upload Your Own CSV Data")
            st.info("💡 Upload a CSV file with survey responses for testing. Must have columns: name, organization, q1_score through q5_score")
            
            uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key='csv_upload')
            
            if uploaded_file:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    
                    # Validate columns
                    required_cols = ['name', 'organization', 'q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
                    missing_cols = [col for col in required_cols if col not in df_upload.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                        st.info("Required columns: name, organization, q1_score, q2_score, q3_score, q4_score, q5_score")
                    else:
                        st.success(f"✓ Valid CSV with {len(df_upload)} rows")
                        st.dataframe(df_upload.head(10))
                        
                        if st.button("📥 Import to Database", type="primary"):
                            # Import each row
                            for idx, row in df_upload.iterrows():
                                responses = []
                                for i in range(1, 6):
                                    score = row[f'q{i}_score']
                                    if pd.notna(score):
                                        # Map score to label
                                        if score >= 4.5:
                                            label = 'Very Satisfied'
                                        elif score >= 3.5:
                                            label = 'Satisfied'
                                        elif score >= 2.5:
                                            label = 'Neutral'
                                        elif score >= 1.5:
                                            label = 'Unsatisfied'
                                        else:
                                            label = 'Very Unsatisfied'
                                    else:
                                        label = 'No Answer'
                                        score = None
                                    
                                    responses.append({
                                        'label': label,
                                        'score': score,
                                        'confidence': 0.95
                                    })
                                
                                save_response(
                                    str(row['name']),
                                    str(row['organization']),
                                    responses
                                )
                            
                            st.success(f"✓ Imported {len(df_upload)} responses!")
                            st.balloons()
                            st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error reading CSV: {str(e)}")
        
        with tab_synthetic:
            st.markdown("#### Generate Synthetic Survey Data")
            st.info("💡 Create fake survey responses for testing ML models and analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                n_responses = st.number_input("Number of responses:", min_value=5, max_value=100, value=20)
                satisfaction_rate = st.slider("Satisfaction rate (%):", 0, 100, 60)
            
            with col2:
                variance = st.slider("Score variance:", 0.0, 2.0, 0.5, 0.1)
                st.caption("Higher variance = more diverse responses")
            
            if st.button("🎲 Generate Synthetic Data", type="primary", key='generate_synthetic_main'):
                # Generate synthetic data
                synthetic_data = []
                
                names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
                        "Kate", "Leo", "Maria", "Nathan", "Olivia", "Peter", "Quinn", "Rachel", "Sam", "Tina"]
                orgs = ["Company A", "Company B", "University", "Government", "NGO", "Startup", "Corporation"]
                
                for i in range(n_responses):
                    # Determine if this should be satisfied or not
                    is_satisfied = np.random.random() < (satisfaction_rate / 100)
                    
                    if is_satisfied:
                        # Generate high scores (3.5-5.0)
                        base_score = np.random.uniform(3.5, 5.0)
                    else:
                        # Generate low scores (1.0-3.5)
                        base_score = np.random.uniform(1.0, 3.5)
                    
                    # Generate 5 question scores with variance
                    scores = []
                    responses = []
                    for _ in range(5):
                        score = np.clip(base_score + np.random.normal(0, variance), 1.0, 5.0)
                        score = round(score * 2) / 2  # Round to nearest 0.5
                        scores.append(score)
                        
                        # Map to label
                        if score >= 4.5:
                            label = 'Very Satisfied'
                        elif score >= 3.5:
                            label = 'Satisfied'
                        elif score >= 2.5:
                            label = 'Neutral'
                        elif score >= 1.5:
                            label = 'Unsatisfied'
                        else:
                            label = 'Very Unsatisfied'
                        
                        responses.append({
                            'label': label,
                            'score': score,
                            'confidence': np.random.uniform(0.85, 0.99)
                        })
                    
                    # Save to database
                    name = np.random.choice(names) + f" {i+1}"
                    org = np.random.choice(orgs)
                    save_response(name, org, responses)
                
                st.success(f"✓ Generated {n_responses} synthetic responses!")
                st.info(f"Distribution: ~{satisfaction_rate}% satisfied, ~{100-satisfaction_rate}% unsatisfied")
                st.balloons()
                st.rerun()
    
    # ========== TAB 3: CLEAN DATA (ENHANCED) ==========
    with tab3:
        st.subheader("🧹 Data Cleaning with Multiple Strategies")
        
        df = get_all_responses()
        
        if len(df) == 0:
            st.info("No data to clean yet.")
        else:
            st.markdown("### Data Quality Check")
            
            score_cols = ['q1_score', 'q2_score', 'q3_score', 'q4_score', 'q5_score']
            
            # Show missing values
            missing_counts = {}
            for col in score_cols:
                missing = df[col].isnull().sum()
                missing_counts[col] = missing
            
            total_missing = sum(missing_counts.values())
            
            col1, col2 = st.columns([2, 1])
            with col1:
                if total_missing > 0:
                    st.warning(f"⚠️ Found {total_missing} missing values")
                    for col, count in missing_counts.items():
                        if count > 0:
                            st.write(f"  - {col}: {count} missing")
                else:
                    st.success("✓ No missing values in database!")
            
            with col2:
                st.metric("Total Missing", total_missing)
                st.metric("Total Responses", len(df))
            
            st.markdown("---")
            st.markdown("### 🎓 Select Imputation Strategy")
            
            strategy = st.selectbox(
                "Choose a strategy:",
                options=list(IMPUTATION_STRATEGIES.keys()),
                format_func=lambda x: IMPUTATION_STRATEGIES[x]['name']
            )
            
            # Show educational info
            show_method_info(IMPUTATION_STRATEGIES, strategy)
            
            # Additional parameters for some strategies
            constant_value = 3
            group_col = None
            
            if strategy == 'constant':
                constant_value = st.number_input("Constant value:", min_value=1, max_value=5, value=3)
            
            if strategy == 'group_mean':
                if 'organization' in df.columns:
                    group_col = 'organization'
                    st.info(f"📊 Will use group means by: {group_col}")
            
            st.markdown("---")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Preview Cleaned Data", type="primary"):
                    st.session_state.cleaned_df = apply_imputation(df, strategy, constant_value, group_col)
                    st.session_state.cleaning_applied = True
                    st.success(f"✓ Applied {IMPUTATION_STRATEGIES[strategy]['name']}!")
            
            with col2:
                if st.button("💾 Save to Database"):
                    if 'cleaned_df' in st.session_state and st.session_state.get('cleaning_applied'):
                        update_responses_with_cleaned(st.session_state.cleaned_df)
                        st.success("✓ Cleaned data saved to database!")
                        st.info("ℹ️ Refresh the page to see updated data everywhere")
                        # Clear the session state
                        del st.session_state.cleaned_df
                        st.session_state.cleaning_applied = False
                        st.rerun()
                    else:
                        st.warning("⚠️ Please preview cleaned data first!")
            
            with col3:
                if 'cleaned_df' in st.session_state and st.session_state.get('cleaning_applied'):
                    csv = st.session_state.cleaned_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        f"cleaned_data_{strategy}.csv",
                        "text/csv",
                        key='download-cleaned'
                    )
            
            # Display comparison
            if 'cleaned_df' in st.session_state and st.session_state.get('cleaning_applied'):
                st.markdown("---")
                st.markdown("### 📊 Data Comparison")
                
                tab_before, tab_after, tab_stats = st.tabs(["Before Cleaning", "After Cleaning", "Statistics"])
                
                with tab_before:
                    st.markdown("#### Original Data (with missing values)")
                    st.dataframe(df[['id', 'name', 'organization'] + score_cols + ['overall_score']], 
                                use_container_width=True)
                    
                    st.markdown("##### Missing Value Summary")
                    for col, count in missing_counts.items():
                        if count > 0:
                            st.write(f"- {col}: {count} missing ({count/len(df)*100:.1f}%)")
                
                with tab_after:
                    st.markdown("#### Cleaned Data (no missing values)")
                    st.dataframe(st.session_state.cleaned_df[['id', 'name', 'organization'] + score_cols + ['overall_score']], 
                                use_container_width=True)
                    
                    # Count remaining missing (should be 0)
                    remaining_missing = st.session_state.cleaned_df[score_cols].isnull().sum().sum()
                    if remaining_missing == 0:
                        st.success("✓ All missing values have been filled!")
                    else:
                        st.warning(f"⚠️ {remaining_missing} missing values remaining")
                
                with tab_stats:
                    st.markdown("#### Statistical Comparison")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Before Cleaning**")
                        st.dataframe(df[score_cols].describe())
                    
                    with col2:
                        st.markdown("**After Cleaning**")
                        st.dataframe(st.session_state.cleaned_df[score_cols].describe())
                    
                    st.markdown("##### Changes Summary")
                    for col in score_cols:
                        before_mean = df[col].mean()
                        after_mean = st.session_state.cleaned_df[col].mean()
                        if pd.notna(before_mean) and pd.notna(after_mean):
                            diff = after_mean - before_mean
                            st.write(f"**{col}**: {before_mean:.2f} → {after_mean:.2f} (change: {diff:+.2f})")
    
    # ========== TAB 4: STATISTICS (ENHANCED) ==========
    with tab4:
        st.subheader("📈 Statistical Analysis")
        
        df = get_all_responses()
        
        if len(df) < 2:
            st.info("Need at least 2 responses for statistical analysis")
        else:
            # Always show saved interpretation if it exists
            saved_interp = get_interpretation('statistics')
            if saved_interp:
                st.success("✅ You have a saved interpretation!")
                with st.expander("📝 View Your Previous Interpretation", expanded=True):
                    st.markdown("**Your Saved Analysis:**")
                    st.info(saved_interp)
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        if st.button("🗑️ Delete", key="clear_stats_top"):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            c.execute("DELETE FROM interpretations WHERE analysis_type='statistics'")
                            conn.commit()
                            conn.close()
                            st.success("✓ Interpretation deleted!")
                            st.rerun()
                    with col2:
                        st.caption("💡 Run analysis again to update your interpretation")
                st.markdown("---")
            
            st.markdown("### 🎓 Select Statistical Methods")
            
            methods = st.multiselect(
                "Choose methods to apply:",
                options=list(STATISTICAL_METHODS.keys()),
                default=['descriptive', 'correlation'],
                format_func=lambda x: STATISTICAL_METHODS[x]['name']
            )
            
            # Show info for each selected method
            for method in methods:
                show_method_info(STATISTICAL_METHODS, method)
            
            if st.button("🔬 Run Statistical Analysis", type="primary"):
                with st.spinner("Analyzing..."):
                    # Clean data first
                    df_clean = apply_imputation(df, 'median')
                    results = perform_statistical_analysis(df_clean, methods)
                    
                    # Display results
                    if 'descriptive' in methods and 'descriptive' in results:
                        st.markdown("### 📊 Descriptive Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Mean", f"{results['descriptive']['mean']:.2f}/5.0")
                        with col2:
                            st.metric("Standard Deviation", f"{results['descriptive']['std']:.2f}")
                        with col3:
                            st.metric("Median", f"{results['descriptive']['median']:.2f}")
                        
                        st.markdown("#### Per-Question Statistics")
                        stats_df = pd.DataFrame(results['descriptive']['per_question']).T
                        st.dataframe(stats_df)
                    
                    if 'normality' in methods and 'normality' in results:
                        st.markdown("### 🔔 Normality Tests")
                        for q, result in results['normality'].items():
                            if 'interpretation' in result:
                                st.write(f"**{q}**: {result['interpretation']}")
                    
                    if 'correlation' in methods and 'correlation' in results:
                        st.markdown("### 🔗 Correlation Matrix")
                        st.dataframe(results['correlation'].style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1))
                    
                    # Visualizations
                    st.markdown("### 📊 Visualizations")
                    fig = plot_basic_stats(df_clean)
                    st.pyplot(fig)
                    
                    # Interpretation box
                    st.markdown("---")
                    st.markdown("### 📝 Add Your Interpretation")
                    st.info("💡 Write your observations, insights, and conclusions from the analysis above. This will be saved and appear at the top when you return!")
                    
                    interpretation = st.text_area(
                        "Your interpretation:",
                        value=get_interpretation('statistics'),
                        height=200,
                        key='stats_interp',
                        placeholder="Example: The overall satisfaction mean is 4.2/5.0, indicating high satisfaction. Q1 and Q2 show strong positive correlation (r=0.85), suggesting content quality and instructor teaching are closely related..."
                    )
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("💾 Save Interpretation", key='save_stats', type="primary"):
                            if interpretation.strip():
                                save_interpretation('statistics', interpretation)
                                st.success("✓ Interpretation saved! Scroll down to see it.")
                                st.balloons()
                                st.rerun()  # Refresh to show in "All Saved Interpretations"
                            else:
                                st.warning("⚠️ Please write an interpretation first!")
                    
                    with col2:
                        if interpretation.strip():
                            word_count = len(interpretation.split())
                            st.caption(f"📝 {word_count} words")
            
            # Always show ALL saved interpretations at bottom (outside of analysis button)
            st.markdown("---")
            st.markdown("### 📚 All Saved Interpretations")
            st.caption("💡 All your saved analysis notes appear here for easy reference")
            
            # Get all saved interpretations
            saved_stats = get_interpretation('statistics')
            
            if saved_stats:
                st.markdown("#### 📊 Statistical Analysis Notes")
                st.info(saved_stats)
                if st.button("🗑️ Delete Statistical Interpretation", key="del_stats_bottom"):
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("DELETE FROM interpretations WHERE analysis_type='statistics'")
                    conn.commit()
                    conn.close()
                    st.success("✓ Deleted!")
                    st.rerun()
            
            # Show ALL ML interpretations
            ml_interps = []
            for model_key in ML_MODELS.keys():
                interp = get_interpretation(f'ml_{model_key}')
                if interp:
                    ml_interps.append((ML_MODELS[model_key]['name'], interp, model_key))
            
            if ml_interps:
                st.markdown("#### 🤖 Machine Learning Notes")
                for idx, (model_name, interp, model_key) in enumerate(ml_interps, 1):
                    with st.expander(f"{idx}. {model_name}", expanded=False):
                        st.info(interp)
                        if st.button(f"🗑️ Delete", key=f"del_ml_{model_key}_bottom"):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            c.execute("DELETE FROM interpretations WHERE analysis_type=?", (f'ml_{model_key}',))
                            conn.commit()
                            conn.close()
                            st.success("✓ Deleted!")
                            st.rerun()
            
            if not saved_stats and not ml_interps:
                st.info("📝 No saved interpretations yet. Run an analysis and save your notes!")
            
            # Always show interpretation history at bottom
            st.markdown("---")
            st.markdown("### 📚 Interpretation History")
            
            # Get all interpretations
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT analysis_type, interpretation, timestamp FROM interpretations WHERE analysis_type = 'statistics' ORDER BY timestamp DESC")
            all_interps = c.fetchall()
            conn.close()
            
            if all_interps:
                st.success(f"✅ {len(all_interps)} interpretation(s) saved")
                
                for idx, (analysis_type, interp, timestamp) in enumerate(all_interps):
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**📝 Saved: {timestamp}**")
                        with col2:
                            if st.button("🗑️ Delete", key=f"del_stat_{idx}"):
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                c.execute("DELETE FROM interpretations WHERE analysis_type=? AND timestamp=?", 
                                         (analysis_type, timestamp))
                                conn.commit()
                                conn.close()
                                st.rerun()
                        
                        st.info(interp)
                        if idx < len(all_interps) - 1:
                            st.markdown("---")
            else:
                st.info("💡 No saved interpretations yet. Run analysis and save your interpretation above!")
    
    # ========== TAB 5: MACHINE LEARNING (ENHANCED) ==========
    with tab5:
        st.subheader("🤖 Machine Learning Analysis")
        
        if not ML_AVAILABLE:
            st.error("ML libraries not available. Install scikit-learn, scipy, matplotlib, seaborn")
            return
        
        df = get_all_responses()
        
        if len(df) < 3:
            st.info(f"Need at least 3 responses for ML analysis. Current responses: {len(df)}")
        else:
            # Always show saved ML interpretations if they exist
            all_ml_interps = []
            for model_key in ML_MODELS.keys():
                interp = get_interpretation(f'ml_{model_key}')
                if interp:
                    all_ml_interps.append((ML_MODELS[model_key]['name'], interp, model_key))
            
            if all_ml_interps:
                st.success(f"✅ You have {len(all_ml_interps)} saved ML interpretation(s)!")
                with st.expander(f"📝 View Your Previous ML Interpretations ({len(all_ml_interps)} saved)", expanded=True):
                    for idx, (model_name, interp, model_key) in enumerate(all_ml_interps):
                        st.markdown(f"### {idx+1}. {model_name}")
                        st.info(interp)
                        if st.button(f"🗑️ Delete", key=f"clear_top_{model_key}"):
                            conn = sqlite3.connect(DB_FILE)
                            c = conn.cursor()
                            c.execute("DELETE FROM interpretations WHERE analysis_type=?", (f'ml_{model_key}',))
                            conn.commit()
                            conn.close()
                            st.success("✓ Interpretation deleted!")
                            st.rerun()
                        if idx < len(all_ml_interps) - 1:
                            st.markdown("---")
                st.markdown("---")
            
            st.markdown("### 🎓 Select Machine Learning Model")
            
            model_type = st.selectbox(
                "Choose a model:",
                options=list(ML_MODELS.keys()),
                format_func=lambda x: ML_MODELS[x]['name']
            )
            
            # Show educational info
            show_method_info(ML_MODELS, model_type)
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner(f"Training {ML_MODELS[model_type]['name']}..."):
                    df_clean = apply_imputation(df, 'median')
                    results = train_ml_model(df_clean, model_type)
                    
                    if 'error' in results:
                        st.error(results['error'])
                    else:
                        # Display results
                        st.markdown("### 📊 Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        with col2:
                            st.metric("Precision", f"{results['precision']:.2%}")
                        with col3:
                            st.metric("Recall", f"{results['recall']:.2%}")
                        with col4:
                            st.metric("F1-Score", f"{results['f1']:.2%}")
                        
                        # Feature importance
                        if 'feature_importance' in results:
                            st.markdown("### 📊 Feature Importance")
                            importance_df = pd.DataFrame(
                                results['feature_importance'].items(),
                                columns=['Question', 'Importance']
                            ).sort_values('Importance', ascending=False)
                            
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.barh(importance_df['Question'], importance_df['Importance'])
                            ax.set_xlabel('Importance')
                            ax.set_title(f'Feature Importance - {ML_MODELS[model_type]["name"]}')
                            st.pyplot(fig)
                            
                            st.dataframe(importance_df)
                        
                        # Confusion Matrix
                        if 'confusion_matrix' in results:
                            st.markdown("### 🎯 Confusion Matrix")
                            cm = np.array(results['confusion_matrix'])
                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title('Confusion Matrix')
                            st.pyplot(fig)
                        
                        # Interpretation box
                        # Interpretation box
                        st.markdown("---")
                        st.markdown("### 📝 Add Your ML Interpretation")
                        st.info(f"💡 Write your observations about the {ML_MODELS[model_type]['name']} results. This will be saved and appear at the top!")
                        
                        ml_interpretation = st.text_area(
                            f"Your interpretation for {ML_MODELS[model_type]['name']}:",
                            value=get_interpretation(f'ml_{model_type}'),
                            height=200,
                            key='ml_interp',
                            placeholder=f"Example: The {ML_MODELS[model_type]['name']} achieved 85% accuracy. Q1 (content) was the most important feature, followed by Q2 (instructor). This suggests..."
                        )
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("💾 Save Interpretation", key='save_ml', type="primary"):
                                if ml_interpretation.strip():
                                    save_interpretation(f'ml_{model_type}', ml_interpretation)
                                    st.success("✓ Interpretation saved! Scroll to see it above.")
                                    st.balloons()
                                    st.rerun()  # Refresh to show in "Previous Interpretations"
                                else:
                                    st.warning("⚠️ Please write an interpretation first!")
                        
                        with col2:
                            if ml_interpretation.strip():
                                word_count = len(ml_interpretation.split())
                                st.caption(f"📝 {word_count} words")
            
            # Always show ML interpretation history at bottom
            st.markdown("---")
            st.markdown("### 📚 All ML Interpretation History")
            
            # Get all ML interpretations
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT analysis_type, interpretation, timestamp FROM interpretations WHERE analysis_type LIKE 'ml_%' ORDER BY timestamp DESC")
            all_ml_interps = c.fetchall()
            conn.close()
            
            if all_ml_interps:
                st.success(f"✅ {len(all_ml_interps)} ML interpretation(s) saved across all models")
                
                for idx, (analysis_type, interp, timestamp) in enumerate(all_ml_interps):
                    model_key = analysis_type.replace('ml_', '')
                    model_name = ML_MODELS.get(model_key, {}).get('name', model_key)
                    
                    with st.container():
                        col1, col2 = st.columns([5, 1])
                        with col1:
                            st.markdown(f"**🤖 {model_name}** - {timestamp}")
                        with col2:
                            if st.button("🗑️ Delete", key=f"del_ml_{idx}"):
                                conn = sqlite3.connect(DB_FILE)
                                c = conn.cursor()
                                c.execute("DELETE FROM interpretations WHERE analysis_type=? AND timestamp=?", 
                                         (analysis_type, timestamp))
                                conn.commit()
                                conn.close()
                                st.rerun()
                        
                        st.info(interp)
                        if idx < len(all_ml_interps) - 1:
                            st.markdown("---")
            else:
                st.info("💡 No saved ML interpretations yet. Train models and save your interpretations above!")

# ============================================================================
# SURVEY PAGE
# ============================================================================

def survey_page():
    """Main survey interface"""
    survey_title = get_setting('survey_title', 'Touchless Satisfaction Survey')
    st.title(f"✋ {survey_title}")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        **Gesture Guide:**
        
        ❤️ Heart = Very Satisfied (5)
        👍 Thumbs Up = Satisfied (4)  
        👎 Thumbs Down = Unsatisfied (2)
        ☝️ Waving = Very Unsatisfied (1)
        ✊ Fist = No Answer
        """)
        
        st.info("Show clear hand gestures for best results!")
    
    # Initialize session
    if 'started' not in st.session_state:
        st.session_state.started = False
        st.session_state.current_q = 0
        st.session_state.responses = []
        st.session_state.completed = False
    
    # Start screen
    if not st.session_state.started:
        st.markdown("## Welcome!")
        st.markdown("Please provide your information to begin the survey.")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name:")
        with col2:
            org = st.text_input("Organization:")
        
        if st.button("🚀 Start Survey", type="primary"):
            st.session_state.name = name or "Anonymous"
            st.session_state.org = org or "N/A"
            st.session_state.started = True
            st.rerun()
        return
    
    # Completed screen
    if st.session_state.completed:
        st.success("✅ Survey Complete!")
        st.balloons()
        
        st.markdown("## Your Responses")
        
        df = pd.DataFrame([{
            'Question': f"Q{i+1}",
            'Response': r['label'],
            'Score': r['score'] or 'N/A',
            'Confidence': f"{r['confidence']:.1%}"
        } for i, r in enumerate(st.session_state.responses)])
        
        st.dataframe(df, use_container_width=True)
        
        scores = [r['score'] for r in st.session_state.responses if r['score']]
        if scores:
            avg_score = sum(scores)/len(scores)
            st.metric("Your Average Score", f"{avg_score:.2f}/5.0")
            
            if avg_score >= 4:
                st.success("🎉 Thank you for your positive feedback!")
            elif avg_score >= 3:
                st.info("👍 Thank you for your feedback!")
            else:
                st.warning("We appreciate your honest feedback and will work to improve!")
        
        if st.button("📝 Submit Another Response"):
            st.session_state.started = False
            st.session_state.current_q = 0
            st.session_state.responses = []
            st.session_state.completed = False
            st.rerun()
        return
    
    # Survey in progress
    current_q = st.session_state.current_q
    total_q = len(SURVEY_QUESTIONS)
    
    st.progress(current_q / total_q, text=f"Question {current_q + 1} of {total_q}")
    
    st.markdown(f"## Question {current_q + 1}")
    st.markdown(f"### {SURVEY_QUESTIONS[current_q]}")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        img_file = st.camera_input("Show your gesture", key=f"cam_{current_q}")
        
        if img_file:
            image = Image.open(img_file)
            
            with st.spinner("Analyzing gesture..."):
                gesture, confidence = simple_predict(image)
            
            info = GESTURE_MAP[gesture]
            
            st.success(f"Detected: {info['emoji']} {info['label']}")
            st.info(f"Confidence: {confidence:.1%}")
            
            if st.button("✅ Confirm", type="primary"):
                st.session_state.responses.append({
                    'label': info['label'],
                    'score': info['score'],
                    'confidence': confidence
                })
                
                if current_q < total_q - 1:
                    st.session_state.current_q += 1
                    st.rerun()
                else:
                    st.session_state.completed = True
                    
                    # Save to database
                    save_response(
                        st.session_state.name,
                        st.session_state.org,
                        st.session_state.responses
                    )
                    
                    st.rerun()
    
    with col2:
        st.markdown("**Gestures:**")
        for g, info in GESTURE_MAP.items():
            st.write(f"{info['emoji']} {info['label']}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize database
    init_database()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["📝 Survey", "🔧 Admin Panel"],
            label_visibility="collapsed"
        )
    
    # Route to appropriate page
    if page == "📝 Survey":
        survey_page()
    else:
        admin_panel()

if __name__ == "__main__":
    main()
