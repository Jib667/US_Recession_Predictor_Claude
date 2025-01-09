import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import fredapi
from datetime import datetime, timedelta

# FRED API key
FRED_API_KEY = 'FRED_API_KEY HERE'


class EnhancedRecessionPredictor:
    def __init__(self):
        self.model = None
        self.fred = fredapi.Fred(api_key=FRED_API_KEY)
        self.feature_order = None

        # Define both indicators and their raw ranges for normalization
        self.indicators = {
            'INDPRO': {
                'fred_id': 'INDPRO',
                'description': 'Industrial Production Index',
                'unit': '2017=100',
                'raw_min': 0,
                'raw_max': 120
            },
            'CPI': {
                'fred_id': 'CPIAUCSL',
                'description': 'Consumer Price Index',
                'unit': '1982-84=100',
                'raw_min': 0,
                'raw_max': 350
            },
            'GDP': {
                'fred_id': 'GDP',
                'description': 'Gross Domestic Product',
                'unit': 'Billions of USD',
                'raw_min': 0,
                'raw_max': 30000
            },
            'Rate': {
                'fred_id': 'FEDFUNDS',
                'description': 'Federal Funds Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            'BBK_Index': {
                'fred_id': 'NFCI',
                'description': 'Financial Conditions Index',
                'unit': 'Index',
                'raw_min': -5,
                'raw_max': 5
            },
            'Housing_Index': {
                'fred_id': 'HOUST',
                'description': 'Housing Starts',
                'unit': 'Thousands of Units',
                'raw_min': 0,
                'raw_max': 2000
            },
            'Price_x': {
                'fred_id': 'CPILFESL',
                'description': 'Core Consumer Price Index',
                'unit': 'Index',
                'raw_min': 0,
                'raw_max': 350
            },
            '3 Mo': {
                'fred_id': 'DTB3',
                'description': '3-Month Treasury Bill Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '6 Mo': {
                'fred_id': 'DTB6',
                'description': '6-Month Treasury Bill Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '1 Yr': {
                'fred_id': 'DGS1',
                'description': '1-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '2 Yr': {
                'fred_id': 'DGS2',
                'description': '2-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '3 Yr': {
                'fred_id': 'DGS3',
                'description': '3-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '5 Yr': {
                'fred_id': 'DGS5',
                'description': '5-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '7 Yr': {
                'fred_id': 'DGS7',
                'description': '7-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '10 Yr': {
                'fred_id': 'DGS10',
                'description': '10-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '20 Yr': {
                'fred_id': 'DGS20',
                'description': '20-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            },
            '30 Yr': {
                'fred_id': 'DGS30',
                'description': '30-Year Treasury Rate',
                'unit': '%',
                'raw_min': 0,
                'raw_max': 20
            }
        }

    def denormalize_value(self, normalized_value, feature):
        """Convert normalized (0-1) value back to raw scale"""
        if feature not in self.indicators:
            raise ValueError(f"No indicator info found for feature: {feature}")

        indicator_info = self.indicators[feature]
        raw_min = indicator_info['raw_min']
        raw_max = indicator_info['raw_max']

        # Convert from 0-1 scale back to raw scale
        return normalized_value * (raw_max - raw_min) + raw_min

    def normalize_value(self, raw_value, feature):
        """Normalize a raw value to 0-1 scale"""
        if feature not in self.indicators:
            raise ValueError(f"No indicator info found for feature: {feature}")

        indicator_info = self.indicators[feature]
        raw_min = indicator_info['raw_min']
        raw_max = indicator_info['raw_max']

        # First normalize to 0-1 scale
        if raw_max == raw_min:
            normalized = 0.5  # Default to middle if range is zero
        else:
            normalized = (raw_value - raw_min) / (raw_max - raw_min)

        # Clip to ensure value is between 0 and 1
        return np.clip(normalized, 0, 1)

    def load_training_data(self, csv_file):
        """Load and process training data"""
        print("Loading training data...")
        df = pd.read_csv(csv_file)

        # Remove '4 Mo' column if present
        if '4 Mo' in df.columns:
            df = df.drop('4 Mo', axis=1)

        # Store feature order
        self.feature_order = [col for col in df.columns
                              if col not in ['Unnamed: 0', 'Recession']]

        # First denormalize the training data to raw values
        denormalized_df = df.copy()
        for feature in self.feature_order:
            if feature in self.indicators:
                denormalized_df[feature] = df[feature].apply(
                    lambda x: self.denormalize_value(x, feature)
                )

        # Then normalize it back using our normalization function
        normalized_df = denormalized_df.copy()
        for feature in self.feature_order:
            if feature in self.indicators:
                normalized_df[feature] = denormalized_df[feature].apply(
                    lambda x: self.normalize_value(x, feature)
                )

        # Add back the Recession column
        normalized_df['Recession'] = df['Recession']

        return normalized_df

    def prepare_data(self, df):
        """Prepare features for model training/prediction"""
        if self.feature_order is None:
            # First time (during training), establish the order
            self.feature_order = [col for col in df.columns
                                  if col not in ['Unnamed: 0', 'Recession']]
        X = df[self.feature_order]  # Use established order
        if 'Recession' in df.columns:
            y = df['Recession']
            return X, y
        return X

    def train(self, df, test_size=0.2, random_state=42):
        """Train the model using historical data"""
        X, y = self.prepare_data(df)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate the model
        evaluation = {
            'train_score': self.model.score(X_train, y_train),
            'test_score': self.model.score(X_test, y_test),
            'cv_scores': cross_val_score(self.model, X_train, y_train, cv=5),
            'y_pred': self.model.predict(X_test),
            'y_test': y_test,
            'roc_auc': roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        }

        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return evaluation, feature_importance

    def fetch_current_data(self):
        """Fetch and normalize current economic indicators from FRED"""
        current_raw = {}
        current_normalized = {}

        print("\nFetching and normalizing current economic indicators...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        if self.feature_order is None:
            raise ValueError("Model must be trained before fetching current data")

        for feature in self.feature_order:
            info = self.indicators[feature]
            try:
                # Fetch data from FRED
                series = self.fred.get_series(
                    info['fred_id'],
                    start_date,
                    end_date
                )

                if len(series) == 0:
                    print(f"Warning: No recent data for {info['description']}")
                    continue

                # Get most recent non-null value
                raw_value = series.dropna().iloc[-1]
                current_raw[feature] = raw_value

                # Normalize using our consistent normalization function
                normalized_value = self.normalize_value(raw_value, feature)
                current_normalized[feature] = normalized_value

                print(f"Normalized {feature}: {raw_value:.2f} -> {normalized_value:.3f}")

            except Exception as e:
                print(f"Warning: Could not fetch {info['description']}: {str(e)}")
                continue

        # Check for missing features
        missing_features = set(self.feature_order) - set(current_normalized.keys())
        if missing_features:
            print(f"Missing features: {missing_features}")
            return None, None

        # Create DataFrame with features in correct order
        df = pd.DataFrame([current_normalized])
        df = df[self.feature_order]  # Ensure correct column order
        return df, current_raw

    def predict_recession_probability(self):
        """Predict recession probability using current economic data"""
        if self.model is None:
            raise ValueError("Model needs to be trained first!")

        # Fetch and normalize current data
        current_data, current_raw = self.fetch_current_data()
        if current_data is None:
            raise ValueError("Could not fetch complete current data")

        # Make prediction
        recession_prob = self.model.predict_proba(current_data)[0, 1]

        # Prepare detailed results
        results = []
        for feature in current_data.columns:
            info = self.indicators[feature]
            results.append({
                'name': info['description'],
                'raw_value': current_raw[feature],
                'normalized_value': current_data[feature].iloc[0],
                'unit': info['unit'],
                'importance': self.model.feature_importances_[list(current_data.columns).index(feature)]
            })

        return {
            'probability': recession_prob,
            'indicators': pd.DataFrame(results).sort_values('importance', ascending=False)
        }


def main():
    try:
        print("Initializing enhanced recession predictor...\n")
        predictor = EnhancedRecessionPredictor()

        print("Loading and analyzing historical data...")
        historical_data = predictor.load_training_data('US_Recession.csv')

        print("\nTraining model...\n")
        evaluation, feature_importance = predictor.train(historical_data)

        print("Model Performance Metrics:")
        print(f"Training Score: {evaluation['train_score']:.3f}")
        print(f"Test Score: {evaluation['test_score']:.3f}")
        print(f"ROC AUC Score: {evaluation['roc_auc']:.3f}")
        print(f"Cross-validation Score: {evaluation['cv_scores'].mean():.3f} "
              f"(+/- {evaluation['cv_scores'].std() * 2:.3f})\n")

        print("Classification Report:")
        print(classification_report(evaluation['y_test'], evaluation['y_pred']))

        print("\nMost Important Economic Indicators:")
        print(feature_importance.to_string(index=False))

        print("\nFetching current economic data and making prediction...")
        prediction = predictor.predict_recession_probability()

        print(f"\nCurrent Recession Probability: {prediction['probability'] * 100:.1f}%")
        print("\nCurrent Economic Indicators:")

        def print_indicator_info(indicator_data):
            print(f"\n{indicator_data['name']}:")
            print(f"  Current Value: {indicator_data['raw_value']:.2f} {indicator_data['unit']}")
            print(f"  Normalized Value: {indicator_data['normalized_value']:.3f} (0-1 scale)")
            print(f"  Importance: {indicator_data['importance'] * 100:.1f}%")

        # Print each indicator
        for _, indicator_data in prediction['indicators'].iterrows():
            print_indicator_info(indicator_data)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please ensure you have all required packages installed:")
        print("pip install pandas numpy scikit-learn fredapi")
        raise


if __name__ == "__main__":
    main()