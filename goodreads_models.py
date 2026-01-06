"""
Goodreads Prediction Models
Tasks:
1. Rating prediction
2. Read prediction
3. Category prediction (text)
"""

import gzip
import math
import numpy as np
import string
from collections import defaultdict, Counter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import gzip
import os
import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def readGz(path):
    """Read gzipped JSON file"""
    for l in gzip.open(path, "rt"):
        yield eval(l)


def readCSV(path):
    """Read CSV file (handles .gz compression)"""
    f = gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
    next(f)  # Skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            yield parts[0], parts[1], float(parts[2])
    f.close()


def Jaccard(s1, s2):
    """Compute Jaccard similarity"""
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer / denom
    return 0


# ============================================================
# TASK 1: RATING PREDICTION
# ============================================================


def predictRatings():
    """ALS-based rating prediction with optimal hyperparameters"""
    # Load training data
    allRatings = [(u, b, r) for u, b, r in readCSV("train_Interactions.csv.gz")]
    ratingsTrain = allRatings

    # Organize data
    ratingsPerUser = defaultdict(list)
    ratingsPerItem = defaultdict(list)
    for u, b, r in ratingsTrain:
        ratingsPerUser[u].append((b, r))
        ratingsPerItem[b].append((u, r))

    users = list(ratingsPerUser.keys())
    items = list(ratingsPerItem.keys())
    alpha = sum(r for _, _, r in ratingsTrain) / len(ratingsTrain)

    # Optimal hyperparameters
    K = 10
    lamb_bias = 2.0
    lamb_latent = 12.0
    n_iterations = 50

    # Initialize parameters
    betaU = defaultdict(float)
    betaI = defaultdict(float)
    np.random.seed(42)
    gammaU = {u: np.random.normal(0, 0.1, K) for u in users}
    gammaI = {i: np.random.normal(0, 0.1, K) for i in items}

    # Training loop
    for iteration in range(n_iterations):
        # Update user biases
        newBetaU = {}
        for u in users:
            numerator = sum(
                r - (alpha + betaI[i] + np.dot(gammaU[u], gammaI[i]))
                for i, r in ratingsPerUser[u]
            )
            newBetaU[u] = numerator / (lamb_bias + len(ratingsPerUser[u]))
        betaU = newBetaU

        # Update item biases
        newBetaI = {}
        for i in items:
            numerator = sum(
                r - (alpha + betaU[u] + np.dot(gammaU[u], gammaI[i]))
                for u, r in ratingsPerItem[i]
            )
            newBetaI[i] = numerator / (lamb_bias + len(ratingsPerItem[i]))
        betaI = newBetaI

        # Update user factors
        newGammaU = {}
        for u in users:
            items_rated = [i for i, _ in ratingsPerUser[u]]
            if not items_rated:
                newGammaU[u] = gammaU[u]
                continue
            A = np.array([gammaI[i] for i in items_rated])
            b = np.array(
                [r - (alpha + betaU[u] + betaI[i]) for i, r in ratingsPerUser[u]]
            )
            newGammaU[u] = np.linalg.lstsq(
                A.T.dot(A) + lamb_latent * np.eye(K), A.T.dot(b), rcond=None
            )[0]
        gammaU = newGammaU

        # Update item factors
        newGammaI = {}
        for i in items:
            users_rated = [u for u, _ in ratingsPerItem[i]]
            if not users_rated:
                newGammaI[i] = gammaI[i]
                continue
            A = np.array([gammaU[u] for u in users_rated])
            b = np.array(
                [r - (alpha + betaU[u] + betaI[i]) for u, r in ratingsPerItem[i]]
            )
            newGammaI[i] = np.linalg.lstsq(
                A.T.dot(A) + lamb_latent * np.eye(K), A.T.dot(b), rcond=None
            )[0]
        gammaI = newGammaI

    # Generate predictions
    with open("predictions_Rating.csv", "w") as f:
        for line in open("pairs_Rating.csv"):
            if line.startswith("userID"):
                f.write(line)
                continue
            u, b = line.strip().split(",")
            pred = alpha + betaU.get(u, 0) + betaI.get(b, 0)
            if u in gammaU and b in gammaI:
                pred += np.dot(gammaU[u], gammaI[b])
            pred = max(1, min(5, pred))
            f.write(f"{u},{b},{pred}\n")


# ============================================================
# TASK 2: READ PREDICTION
# ============================================================
import numpy as np
from collections import defaultdict
import random


def predictRead():
    """
    Improved BPR with better threshold and cold-start handling
    """
    # ============================================================
    # LOAD DATA
    # ============================================================
    userBooks = defaultdict(set)
    bookUsers = defaultdict(set)
    bookPopularity = defaultdict(int)
    allBooks = set()
    allUsers = set()

    for user, book, _ in readCSV("train_Interactions.csv.gz"):
        userBooks[user].add(book)
        bookUsers[book].add(user)
        bookPopularity[book] += 1
        allBooks.add(book)
        allUsers.add(user)

    active_users = [u for u in userBooks if len(userBooks[u]) > 0]
    users = list(allUsers)
    books = list(allBooks)

    userToIdx = {u: i for i, u in enumerate(users)}
    bookToIdx = {b: i for i, b in enumerate(books)}

    # ============================================================
    # CALCULATE BASELINE STATISTICS
    # ============================================================
    totalInteractions = sum(len(userBooks[u]) for u in users)
    avgBooksPerUser = totalInteractions / len(users)
    popularityThreshold = np.percentile(list(bookPopularity.values()), 50)

    # ============================================================
    # OPTIMIZED HYPERPARAMETERS
    # ============================================================
    K = 10
    learning_rate = 0.08
    reg_lambda = 0.01
    n_iterations = 20
    n_samples = 20000

    # ============================================================
    # INITIALIZE
    # ============================================================
    np.random.seed(42)
    userFactors = np.random.normal(0, 0.01, (len(users), K))
    bookFactors = np.random.normal(0, 0.01, (len(books), K))
    userBias = np.zeros(len(users))
    bookBias = np.zeros(len(books))

    # ============================================================
    # BPR TRAINING
    # ============================================================
    for iteration in range(n_iterations):
        for _ in range(n_samples):
            u = random.choice(active_users)
            u_idx = userToIdx[u]

            # Positive sample
            i = random.choice(list(userBooks[u]))
            i_idx = bookToIdx[i]

            # Negative sample
            j_idx = random.randint(0, len(books) - 1)
            while books[j_idx] in userBooks[u]:
                j_idx = random.randint(0, len(books) - 1)

            # Compute scores
            x_ui = (
                userBias[u_idx]
                + bookBias[i_idx]
                + np.dot(userFactors[u_idx], bookFactors[i_idx])
            )
            x_uj = (
                userBias[u_idx]
                + bookBias[j_idx]
                + np.dot(userFactors[u_idx], bookFactors[j_idx])
            )
            x_uij = np.clip(x_ui - x_uj, -10, 10)

            # Gradient
            sigmoid = 1.0 / (1.0 + np.exp(-x_uij))
            gradient = 1 - sigmoid

            # Updates
            userFactors[u_idx] += learning_rate * (
                gradient * (bookFactors[i_idx] - bookFactors[j_idx])
                - reg_lambda * userFactors[u_idx]
            )
            bookFactors[i_idx] += learning_rate * (
                gradient * userFactors[u_idx] - reg_lambda * bookFactors[i_idx]
            )
            bookFactors[j_idx] += learning_rate * (
                -gradient * userFactors[u_idx] - reg_lambda * bookFactors[j_idx]
            )
            userBias[u_idx] += learning_rate * (gradient - reg_lambda * userBias[u_idx])
            bookBias[i_idx] += learning_rate * (gradient - reg_lambda * bookBias[i_idx])
            bookBias[j_idx] += learning_rate * (
                -gradient - reg_lambda * bookBias[j_idx]
            )

    # ============================================================
    # IMPROVED THRESHOLD CALCULATION
    # ============================================================
    # Calculate positive rate in training data
    positive_rate = totalInteractions / (len(users) * len(books))

    # Sample scores to find appropriate threshold
    pos_scores = []
    neg_scores = []

    sample_users = random.sample(active_users, min(1000, len(active_users)))
    for u in sample_users:
        u_idx = userToIdx[u]

        # Sample positive items
        if len(userBooks[u]) > 0:
            for b in random.sample(list(userBooks[u]), min(5, len(userBooks[u]))):
                b_idx = bookToIdx[b]
                score = (
                    userBias[u_idx]
                    + bookBias[b_idx]
                    + np.dot(userFactors[u_idx], bookFactors[b_idx])
                )
                pos_scores.append(score)

        # Sample negative items
        for _ in range(5):
            b_idx = random.randint(0, len(books) - 1)
            if books[b_idx] not in userBooks[u]:
                score = (
                    userBias[u_idx]
                    + bookBias[b_idx]
                    + np.dot(userFactors[u_idx], bookFactors[b_idx])
                )
                neg_scores.append(score)

    # Set threshold between positive and negative distributions
    if pos_scores and neg_scores:
        threshold = (np.mean(pos_scores) + np.mean(neg_scores)) / 2
    else:
        threshold = 0.0

    # ============================================================
    # GENERATE PREDICTIONS WITH SMART FALLBACK
    # ============================================================
    with open("predictions_Read.csv", "w") as f:
        for line in open("pairs_Read.csv"):
            if line.startswith("userID"):
                f.write(line)
                continue

            u, b = line.strip().split(",")
            u_idx = userToIdx.get(u)
            b_idx = bookToIdx.get(b)

            if u_idx is not None and b_idx is not None:
                # BPR score
                bpr_score = (
                    userBias[u_idx]
                    + bookBias[b_idx]
                    + np.dot(userFactors[u_idx], bookFactors[b_idx])
                )

                # Popularity boost for very popular books
                pop_boost = 0.2 if bookPopularity.get(b, 0) > popularityThreshold else 0

                final_score = bpr_score + pop_boost
                pred = 1 if final_score > threshold else 0

            elif b_idx is not None:
                # Cold start: user unknown, use book popularity
                pred = 1 if bookPopularity.get(b, 0) > popularityThreshold else 0

            elif u_idx is not None:
                # Cold start: book unknown, use user's average activity
                user_activity = len(userBooks.get(u, []))
                pred = 1 if user_activity > avgBooksPerUser else 0

            else:
                # Both unknown: use global popularity
                pred = 1 if bookPopularity.get(b, 0) > popularityThreshold else 0

            f.write(f"{u},{b},{pred}\n")


def readCSV(path):
    import gzip

    f = gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
    next(f)
    for line in f:
        parts = line.strip().split(",")
        if len(parts) >= 3:
            yield parts[0], parts[1], float(parts[2])
    f.close()


if __name__ == "__main__":
    predictRead()


# ============================================================
# TASK 3: CATEGORY PREDICTION - OPTIMIZED VERSION
# ============================================================


"""
TASK 3: CATEGORY PREDICTION
----------------------------
Approach: TF-IDF with Bigrams for Text Classification
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np


def optimizedFeatures(data, vectorizer=None, fit=True):
    """
    Extract improved features using TF-IDF, unigrams+bigrams, larger vocabulary
    """
    texts = [d["review_text"] for d in data]

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=2500,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",  # ← ADD THIS
            lowercase=True,
            strip_accents="unicode",
            token_pattern=r"\b[a-zA-Z]{2,}\b",
        )

    if fit:
        X = vectorizer.fit_transform(texts).toarray()
    else:
        X = vectorizer.transform(texts).toarray()

    return X, vectorizer


def predictCategory():
    """
    Predict book category using TF-IDF and Logistic Regression
    """
    # Load training data
    data = []
    for d in readGz("train_Category.json.gz"):
        data.append(d)

    # Extract features and train model
    X_train, vectorizer = optimizedFeatures(data, fit=True)
    y_train = [d["genreID"] for d in data]

    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    # Load test data and generate predictions
    data_test = []
    for d in readGz("test_Category.json.gz"):
        data_test.append(d)

    X_test, _ = optimizedFeatures(data_test, vectorizer=vectorizer, fit=False)
    pred_test = model.predict(X_test)

    # Write predictions
    predictions = open("predictions_Category.csv", "w")
    pos = 0

    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue

        u, b = l.strip().split(",")
        predictions.write(u + "," + b + "," + str(pred_test[pos]) + "\n")
        pos += 1

    predictions.close()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GOODREADS BOOK REVIEW PREDICTION - ASSIGNMENT 1")
    print("=" * 60 + "\n")

    # Task 1: Rating Prediction
    predictRatings()

    # Task 2: Read Prediction
    predictRead()

    # Task 3: Category Prediction
    predictCategory()

    print("=" * 60)
    print("ALL TASKS COMPLETED!")
    print("=" * 60)
    print("\nGenerated files:")

    output_files = [
        "predictions_Rating.csv",
        "predictions_Read.csv",
        "predictions_Category.csv",
    ]
