import csv
import random
import re
import sys
import os
import argparse
from typing import Dict, Tuple, Union

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from numpy.typing import ArrayLike
except ModuleNotFoundError:
    # There can be many possible types, but we recommend you to use types below.
    ArrayLike = Union[list, tuple, np.ndarray]


"""
# Bag-of-Words Classification with scikit-learn
In this task, you will implement the Bag-of-Words model and text classification in Python.
Implement two methods:
- `preprocess_and_split_to_tokens(sentences) -> tokens_per_sentence`
- `create_bow(sentences, vocab, msg_prefix) -> (vocab, bow_array)`
## Instruction
* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.
* Do not use the bag-of-words function implemented in scikit-learn.
* Before submit your code in KLMS, please change the name of the file to your student id (e.g., 2019xxxx.py).
* Functionality and prediction accuracy for unknown test samples (i.e., we do not give them to you) will be your grade.
* For functionality, we will run unit tests of `preprocess_and_split_to_tokens` and `create_bow`.
* For prediction accuracy, if it is on par with the score of TA, you will get a perfect score.
* TA's validation accuracy is 0.861.
* See https://scikit-learn.org/stable/modules/classes.html for more information.
"""


def _download_dataset(size=10000):
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        print("Download: {}".format(file_path))
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
    else:
        print("Already exist: {}".format(file_path))


def _get_review_data(path, num_samples, train_test_ratio=0.8):
    """Do not modify the code in this function."""
    _download_dataset()
    print("Load Data at {}".format(path))
    reviews, sentiments = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            reviews.append(line["review"])
            sentiments.append(int(line["sentiment"]))

    # Data shuffle
    random.seed(42)
    zipped = list(zip(reviews, sentiments))
    random.shuffle(zipped)
    reviews, sentiments = zip(*(zipped[:num_samples]))
    reviews, sentiments = np.asarray(reviews), np.asarray(sentiments)

    # Train/test split
    num_data, num_train = len(sentiments), int(len(sentiments) * train_test_ratio)
    return (reviews[:num_train], sentiments[:num_train]), (reviews[num_train:], sentiments[num_train:])


def _get_example_of_errors(texts_to_analyze, preds_to_analyze, labels_to_analyze):
    """Do not modify the code in this function."""
    texts_to_analyze = texts_to_analyze[np.random.permutation(len(texts_to_analyze))]
    correct = texts_to_analyze[preds_to_analyze == labels_to_analyze]
    wrong = texts_to_analyze[preds_to_analyze != labels_to_analyze]
    print("\n[Correct Sample Examples]")
    for line in correct[:5]:
        print("\t- {}".format(line))
    print("\n[Wrong Sample Examples]")
    for line in wrong[:5]:
        print("\t- {}".format(line))


def preprocess_and_split_to_tokens(sentences: ArrayLike) -> ArrayLike:
    """
    :param sentences: (ArrayLike) ArrayLike objects of strings.
        e.g., ["I like apples", "I love python3"]
    You can choose the level of pre-processing by yourself.
    The easiest way to start is lowering the case (str.lower).
    :return: ArrayLike objects of ArrayLike objects of tokens.
        e.g., [["I", "like", "apples"], ["I", "love", "python3"]]
    """
    ans = []
    for i in range(len(sentences)):
        ans.append([])
        for sentence in sentences[i].split("<br />"):
            new_sentence = ""
            words = list(sentence)
            for j in range(len(words)):
                if words[j].isalpha():
                    new_sentence += words[j].lower()
                else:
                    new_sentence += ' '
                    new_sentence += words[j]
            for word in new_sentence.split(' '):
                if len(word) > 0:
                    ans[i].append(word)

    return ans
            
    raise NotImplementedError


def create_bow(sentences: ArrayLike, vocab: Dict[str, int] = None,
               msg_prefix="\n") -> Tuple[Dict[str, int], ArrayLike]:
    """Make the Bag-of-Words model from the sentences, return (vocab, bow_array)
        vocab: dictionary of (token, index of BoW representation) pair. If None, construct vocab first.
        bow_array: ArrayLike objects of BoW representation, the shape of which is [#sentence_list, #vocab]
    :param sentences: (ArrayLike): ArrayLike objects of strings
        e.g., ["I like apples", "I love python3"]
    :param vocab: (dict, optional)
        e.g., {"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4}
    :param msg_prefix: (str, optional)
    :return: Tuple[dict, ArrayLike]
        e.g., ({"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4},
                [[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
    """
    tokens_per_sentence = preprocess_and_split_to_tokens(sentences)

    if vocab is None:
        print("{} Vocab construction".format(msg_prefix))

        vocab = dict()
        for words in tokens_per_sentence:
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    
        #raise NotImplementedError

    ans = []
    for i in range(len(tokens_per_sentence)):
        ans.append([0 for x in range(len(vocab))])
        for word in tokens_per_sentence[i]:
            if word in vocab:
                ans[i][vocab[word]] += 1

    return (vocab, ans)

    print("{} Bow construction".format(msg_prefix))
    raise NotImplementedError


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/review_10k.csv", num_samples=num_samples)
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))

    # Create bow representation of train set
    my_vocab, train_bows = create_bow(train_xs, msg_prefix="\n[Train]")
    assert isinstance(my_vocab, dict)
    if verbose:
        print("\n[Vocab]: {} words".format(len(my_vocab)))

    # You can see hyper-parameters that can be tuned in the document below.
    #   https://scikit-learn.org/stable/modules/classes.html.
    clf = LogisticRegression(verbose=1, solver="liblinear")
    clf.fit(train_bows, train_ys)
    assert hasattr(clf, "predict")

    # Create bow representation of validation set
    _, val_bows = create_bow(val_xs, vocab=my_vocab, msg_prefix="\n[Validation]")

    # Evaluation
    val_preds = clf.predict(val_bows)
    val_accuracy = accuracy_score(val_ys, val_preds)
    if verbose:
        print("\n[Validation] Accuracy: {}".format(val_accuracy))
        _get_example_of_errors(val_xs, val_preds, val_ys)

    # Grading: Do not modify below lines.
    if test_xs is not None:
        _, test_bows = create_bow(test_xs, vocab=my_vocab, msg_prefix="\n[Test]")
        test_preds = clf.predict(test_bows)
        return {"clf": clf, "val_accuracy": val_accuracy, "test_accuracy": accuracy_score(test_ys, test_preds)}
    else:
        return {"clf": clf}


if __name__ == '__main__':
    # Usage $ python bow_classification_with_sklearn.py --num-samples 10000 --verbose True
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", default=10000, type=int)
    parser.add_argument("--verbose", default=True, type=bool)
    args = parser.parse_args()

    run(
        num_samples=args.num_samples,
        verbose=args.verbose,
    )
