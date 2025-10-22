"""Enhanced Tabular Neural Network Classifier with Categorical Embeddings."""

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TabularNNClassifier(ClassifierMixin, BaseEstimator):
    """Enhanced neural network for binary and multiclass classification with categorical embeddings.

    This classifier automatically detects whether the problem is binary or multiclass
    and adjusts its architecture and loss function accordingly.

    Parameters
    ----------
    hidden_sizes : list of int, default=[200, 100]
        Sizes of hidden layers.
    dropout : float, default=0.1
        Dropout probability.
    learning_rate : float, default=0.001
        Learning rate for optimizer.
    batch_size : int, default=256
        Training batch size.
    epochs : int, default=100
        Number of training epochs.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    activation : str, default='relu'
        Activation function ('relu' or 'elu').
    optimizer : str, default='adam'
        Optimizer type ('adam' or 'adamw').
    random_state : int, default=None
        Random seed.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_sizes=None,
        dropout=0.1,
        learning_rate=0.001,
        batch_size=256,
        epochs=100,
        weight_decay=1e-5,
        activation="relu",
        optimizer="adam",
        random_state=None,
    ):
        self.hidden_sizes = hidden_sizes if hidden_sizes is not None else [200, 100]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.activation = activation
        self.optimizer = optimizer
        self.random_state = random_state

    def _detect_categorical_columns(self, X):
        """Detect categorical columns from DataFrame."""
        if isinstance(X, pd.DataFrame):
            cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            return cat_cols, num_cols
        else:
            return [], list(range(X.shape[1]))

    def fit(self, X, y):
        """Fit the model."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Store classes and determine if binary or multiclass
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.is_binary_ = self.n_classes_ == 2

        if self.n_classes_ < 2:
            raise ValueError(f"Classifier expects at least 2 classes, got {self.n_classes_}")

        # For multiclass, encode target labels to 0, 1, 2, ...
        if not self.is_binary_:
            self.target_encoder_ = LabelEncoder()
            y_encoded = self.target_encoder_.fit_transform(y)

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Detect categorical and numerical columns
        self.cat_cols_, self.num_cols_ = self._detect_categorical_columns(X)

        # Encode categorical features
        self.label_encoders_ = {}
        self.embedding_sizes_ = {}

        X_cat_encoded = []
        for col in self.cat_cols_:
            le = LabelEncoder()
            X_col = X[col].fillna("__NAN__")
            encoded = le.fit_transform(X_col)
            self.label_encoders_[col] = le

            cardinality = len(le.classes_)
            emb_dim = min(50, max(3, (cardinality + 1) // 2))
            self.embedding_sizes_[col] = (cardinality, emb_dim)

            X_cat_encoded.append(encoded)

        # Enhanced missing value handling for numerical features
        self.num_medians_ = {}
        self.missing_indicators_ = []

        X_num_list = []
        for col in self.num_cols_:
            col_data = X[col]
            is_missing = col_data.isna()

            median_val = col_data.median()
            self.num_medians_[col] = median_val if not pd.isna(median_val) else 0.0

            filled_data = col_data.fillna(self.num_medians_[col])
            X_num_list.append(filled_data.values)

            if is_missing.any():
                self.missing_indicators_.append(col)
                X_num_list.append(is_missing.astype(np.float32).values)

        X_num = np.column_stack(X_num_list).astype(np.float32) if X_num_list else np.zeros((len(X), 0))
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Build model
        self.model_ = self._build_model()

        # Convert to tensors
        X_cat_tensor = torch.LongTensor(X_cat)
        X_num_tensor = torch.FloatTensor(X_num)

        if self.is_binary_:
            # Binary: labels 0 or 1
            y_array = y.values if isinstance(y, pd.Series) else y
            y_binary = (y_array == self.classes_[1]).astype(np.float32)
            y_tensor = torch.FloatTensor(y_binary).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            # Multiclass: labels 0, 1, 2, ...
            y_tensor = torch.LongTensor(y_encoded)
            criterion = nn.CrossEntropyLoss()

        # Training
        dataset = TensorDataset(X_cat_tensor, X_num_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Select optimizer
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
            )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

        self.model_.train()
        for _epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_cat_batch, X_num_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model_(X_cat_batch, X_num_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)

        return self

    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self, "model_")

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])

        # Encode categorical features
        X_cat_encoded = []
        for col in self.cat_cols_:
            le = self.label_encoders_[col]
            X_col = X[col].fillna("__NAN__")
            encoded = np.array([le.transform([val])[0] if val in le.classes_ else 0 for val in X_col])
            X_cat_encoded.append(encoded)

        # Prepare numerical features
        X_num_list = []
        for col in self.num_cols_:
            col_data = X[col]
            is_missing = col_data.isna()

            filled_data = col_data.fillna(self.num_medians_[col])
            X_num_list.append(filled_data.values)

            if col in self.missing_indicators_:
                X_num_list.append(is_missing.astype(np.float32).values)

        X_num = np.column_stack(X_num_list).astype(np.float32) if X_num_list else np.zeros((len(X), 0))
        X_cat = np.column_stack(X_cat_encoded) if X_cat_encoded else np.zeros((len(X), 0), dtype=np.int64)

        # Convert to tensors and predict
        X_cat_tensor = torch.LongTensor(X_cat)
        X_num_tensor = torch.FloatTensor(X_num)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_cat_tensor, X_num_tensor)

            if self.is_binary_:
                # Binary classification
                probs_class1 = torch.sigmoid(logits).numpy().flatten()
                probs_class0 = 1 - probs_class1
                return np.column_stack([probs_class0, probs_class1])
            else:
                # Multiclass classification
                probs = torch.softmax(logits, dim=1).numpy()
                return probs

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        y_pred_encoded = np.argmax(proba, axis=1)

        if self.is_binary_:
            return self.classes_[y_pred_encoded]
        else:
            return self.target_encoder_.inverse_transform(y_pred_encoded)

    def _build_model(self):
        """Build the neural network."""
        n_num_features = len(self.num_cols_) + len(self.missing_indicators_)

        # Determine output size: 1 for binary, n_classes for multiclass
        output_size = 1 if self.is_binary_ else self.n_classes_

        return TabularNNClassificationModel(
            cat_cols=self.cat_cols_,
            embedding_sizes=self.embedding_sizes_,
            n_num_features=n_num_features,
            output_size=output_size,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            activation=self.activation,
        )


class TabularNNClassificationModel(nn.Module):
    """PyTorch model for binary and multiclass classification."""

    def __init__(
        self, cat_cols, embedding_sizes, n_num_features, output_size, hidden_sizes, dropout, activation="relu"
    ):
        super().__init__()

        # Embeddings for categorical features
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_classes, emb_dim) for num_classes, emb_dim in embedding_sizes.values()]
        )

        # Calculate input dimension
        total_emb_dim = sum(emb_dim for _, emb_dim in embedding_sizes.values())
        input_dim = total_emb_dim + n_num_features

        # Select activation function
        if activation == "elu":
            activation_fn = nn.ELU()
        else:
            activation_fn = nn.ReLU()

        # Build hidden layers with batch normalization
        layers = []
        prev_size = input_dim
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    activation_fn,
                    nn.Dropout(dropout),
                ]
            )
            prev_size = hidden_size

        # Output layer: 1 unit for binary, n_classes for multiclass
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, X_cat, X_num):
        """Forward pass."""
        # Embed categorical features
        if X_cat.shape[1] > 0:
            embedded = [emb(X_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_features = torch.cat(embedded, dim=1)
        else:
            cat_features = torch.empty(X_cat.shape[0], 0)

        # Concatenate with numerical features
        x = torch.cat([cat_features, X_num], dim=1)

        return self.network(x)
