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

from typing import Iterable, Optional
import logging
import pandas as pd
from mlchem.helper import coerce_log_level


logger = logging.getLogger(__name__)


def _log_if_verbose(verbose: bool, log_level: int, msg: str, *args) -> None:
    if verbose:
        logger.log(log_level, msg, *args)


def check_class_balance(
    y_train: Iterable,
    verbose: bool = False,
    log_level: int | str = logging.INFO,
) -> None:
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
    resolved_log_level = coerce_log_level(log_level)
    _log_if_verbose(
        verbose,
        resolved_log_level,
        'CLASS BALANCE [0]: %d [1]: %d (%.2f/%.2f)',
        zero_class_training,
        one_class_training,
        zero_ratio,
        one_ratio,
    )


def undersample(
    train_set: pd.DataFrame,
    test_set: pd.DataFrame,
    class_column: str,
    desired_proportion_majority: float,
    add_dropped_to_test: bool = False,
    random_seed: Optional[int] = 1,
    verbose: bool = False,
    log_level: int | str = logging.INFO,
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

verbose : bool, default=False
    If True, emit class-balance and sampling diagnostics through logging.

log_level : int or str, default=logging.INFO
    Logging level used when `verbose=True`.

Returns
-------
tuple of pandas.DataFrame
    The undersampled training set and the updated test set.
"""

    import random
    resolved_log_level = coerce_log_level(log_level)

    if not 0 < desired_proportion_majority < 1:
        raise ValueError(
            "'desired_proportion_majority' must be strictly between 0 and 1."
        )

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
    _log_if_verbose(verbose, resolved_log_level, 'Samples to remove: %d', cycles)

    if random_seed is not None:
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
    _log_if_verbose(
        verbose,
        resolved_log_level,
        'CLASS BALANCE [0]: %d [1]: %d (%.2f/%.2f)',
        zero_class_training_undersampled,
        one_class_training_undersampled,
        zero_ratio_undersampled,
        one_ratio_undersampled,
    )

    if add_dropped_to_test:
        to_add = train_set.loc[to_drop_indices]
        test_set_oversampled = pd.concat([test_set, to_add], axis=0)
        return train_set_undersampled, test_set_oversampled
    else:
        return train_set_undersampled, test_set
