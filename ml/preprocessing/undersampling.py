from typing import Iterable, Optional
import pandas as pd


def check_class_balance(y_train: Iterable) -> None:
    """
Check and print the class distribution in training labels.

Parameters
----------
y_train : Iterable
    Training target values.

Returns
-------
None
"""

    zero_class_training = y_train.count(0)
    one_class_training = y_train.count(1)
    total = zero_class_training + one_class_training
    zero_ratio = zero_class_training / total
    one_ratio = 1 - zero_ratio
    print(f'CLASS BALANCE\n\n\n\n[0]: {zero_class_training}  [1]: '
          f'{one_class_training}  ({zero_ratio:.2f}/{one_ratio:.2f})')


def undersample(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    class_column: str,
    desired_proportion_majority: float,
    add_dropped_to_test: bool = False,
    random_seed: Optional[int] = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
Undersample the majority class in a training set to achieve a desired 
class balance.

Parameters
----------
train_set : pandas.DataFrame
    The training dataset.

test_set : pandas.DataFrame
    The test dataset.

class_column : str
    Name of the column containing class labels.

desired_proportion_majority : float
    Desired proportion of the majority class in the training set.

add_dropped_to_test : bool, default=False
    Whether to add the dropped samples to the test set.

random_seed : int, optional
    Random seed for reproducibility.

Returns
-------
tuple of pandas.DataFrame
    The undersampled training set and the updated test set.
"""

    import random

    zero_class_training = train_set[class_column].value_counts().get(0, 0)
    one_class_training = train_set[class_column].value_counts().get(1, 0)

    # Determine the minority and majority classes
    if zero_class_training > one_class_training:
        minority_class = one_class_training
        majority_class = zero_class_training
        majority_class_label = 0
    else:
        minority_class = zero_class_training
        majority_class = one_class_training
        majority_class_label = 1

    cycles = majority_class - int(minority_class *
                                  desired_proportion_majority /
                                  (1 - desired_proportion_majority)
                                  )
    print('Samples to remove:', cycles)

    if random_seed:
        random.seed(random_seed)
    to_drop_indices = random.sample(
        list(train_set[train_set[class_column] == majority_class_label].index),
        cycles
    )

    train_set_undersampled = train_set.drop(index=to_drop_indices)
    y_train_undersampled = train_set_undersampled[class_column].tolist()

    # Check class balance after undersampling
    zero_class_training_undersampled = y_train_undersampled.count(0)
    one_class_training_undersampled = y_train_undersampled.count(1)
    total_undersampled = zero_class_training_undersampled + \
        one_class_training_undersampled
    zero_ratio_undersampled = zero_class_training_undersampled / \
        total_undersampled
    one_ratio_undersampled = 1 - zero_ratio_undersampled
    print(f'CLASS BALANCE\n\n\n\n[0]: {zero_class_training_undersampled}  [1]:'
          f' {one_class_training_undersampled} ({zero_ratio_undersampled:.2f}/'
          f' {one_ratio_undersampled:.2f})')

    if add_dropped_to_test:
        to_add = train_set.loc[to_drop_indices]
        test_set_oversampled = pd.concat([test_set, to_add], axis=0)
        return train_set_undersampled, test_set_oversampled
    else:
        return train_set_undersampled, test_set
