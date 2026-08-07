# mlchem - cheminformatics library
# Copyright © 2025 as Unilever Global IP Limited

# Redistribution and use in source and binary forms, with or without modification,
# are permitted under the terms of the BSD-3 License, provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in
#        the documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# You should have received a copy of the BSD-3 License along with mlchem.
# If not, see https://interoperable-europe.ec.europa.eu/licence/bsd-3-clause-new-or-revised-license .
# It is the responsibility of mlchem users to familiarise themselves with all dependencies and their associated licenses.

from typing import Iterable, Optional, Literal
import logging
import pandas as pd
from mlchem.helper import coerce_log_level


logger = logging.getLogger(__name__)


def _configure_module_logging(level: int) -> None:
    """Configure module logger to emit at the specified level."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.setLevel(level)


def check_class_balance(
    y_train: Iterable,
    log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
) -> None:
    """
Check and log the class distribution in training labels.

Parameters
----------
y_train : Iterable
    Training target values.

log_level : int or str, default=logging.INFO
    Logging level for output.
"""

    counts = pd.Series(list(y_train)).value_counts()
    if counts.empty:
        raise ValueError("'y_train' must contain at least one class label.")

    total = int(counts.sum())
    balance_parts = [
        f"[{label}]: {int(count)} ({(count / total):.2f})"
        for label, count in counts.items()
    ]
    resolved_log_level = coerce_log_level(log_level)
    _configure_module_logging(resolved_log_level)
    logger.log(resolved_log_level, 'CLASS BALANCE %s',
        ' '.join(balance_parts),
    )


def undersample(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    class_column: str,
    desired_proportion_majority: float,
    add_dropped_to_test: bool = False,
    random_seed: Optional[int] = 1,
    log_level: int | str | Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = logging.INFO,
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

log_level : int or str, default=logging.INFO
    Logging level threshold for emitted diagnostics.

Returns
-------
tuple of pandas.DataFrame
    The undersampled training set and the updated test set.
"""

    import random
    resolved_log_level = coerce_log_level(log_level)
    _configure_module_logging(resolved_log_level)

    if not 0 < desired_proportion_majority < 1:
        raise ValueError(
            "'desired_proportion_majority' must be strictly between 0 and 1."
        )

    class_counts = train_set[class_column].value_counts()
    if len(class_counts) != 2:
        raise ValueError(
            "'undersample' supports exactly two classes. "
            "Use binary labels for undersampling."
        )

    majority_class_label = class_counts.idxmax()
    majority_class = int(class_counts.loc[majority_class_label])
    minority_class = int(class_counts.min())

    target_majority_count = int(
        minority_class * desired_proportion_majority /
        (1 - desired_proportion_majority)
    )
    cycles = majority_class - target_majority_count
    if cycles < 0:
        raise ValueError(
            "'desired_proportion_majority' implies majority-class growth. "
            "Use oversampling for this target."
        )
    if cycles > majority_class:
        raise ValueError(
            "Computed majority-class removals exceed available samples."
        )
    logger.log(resolved_log_level, 'Samples to remove: %d', cycles)

    if random_seed is not None:
        random.seed(random_seed)
    to_drop_indices = random.sample(
        list(train_set[train_set[class_column] == majority_class_label].index),
        cycles
    )

    train_set_undersampled = train_set.drop(index=to_drop_indices)
    undersampled_counts = train_set_undersampled[class_column].value_counts()
    total_undersampled = int(undersampled_counts.sum())
    balance_parts = [
        f"[{label}]: {int(count)} ({(count / total_undersampled):.2f})"
        for label, count in undersampled_counts.items()
    ]
    logger.log(resolved_log_level, 'CLASS BALANCE %s',
        ' '.join(balance_parts),
    )

    if add_dropped_to_test:
        to_add = train_set.loc[to_drop_indices]
        test_set_oversampled = pd.concat([test_set, to_add], axis=0)
        return train_set_undersampled, test_set_oversampled
    else:
        return train_set_undersampled, test_set
