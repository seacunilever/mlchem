import pytest
import pandas as pd
from mlchem.ml.preprocessing.undersampling import check_class_balance, undersample

@pytest.fixture
def sample_data():
    train_set = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'class': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    test_set = pd.DataFrame({
        'feature1': [11, 12, 13, 14, 15],
        'class': [0, 1, 0, 1, 0]
    })
    return train_set, test_set

def test_check_class_balance(capsys,sample_data):
    train_set, test_set = sample_data
    y_train = train_set['class'].values.tolist()
    check_class_balance(y_train)
    captured = capsys.readouterr()
    assert "CLASS BALANCE" in captured.out
    assert "[0]: 2  [1]: 8  (0.20/0.80)" in captured.out

def test_undersample(sample_data):
    train_set, test_set = sample_data
    undersampled_train_set, undersampled_test_set = undersample(
        train_set=train_set,
        test_set=test_set,
        class_column='class',
        desired_proportion_majority=0.5,
        add_dropped_to_test=False,
        random_seed=1
    )
    assert len(undersampled_train_set) == 4
    assert len(undersampled_test_set) == 5

def test_undersample_add_dropped_to_test(sample_data):
    train_set, test_set = sample_data
    undersampled_train_set, undersampled_test_set = undersample(
        train_set=train_set,
        test_set=test_set,
        class_column='class',
        desired_proportion_majority=0.5,
        add_dropped_to_test=True,
        random_seed=1
    )
    assert len(undersampled_train_set) == 4
    assert len(undersampled_test_set) == 11

if __name__ == "__main__":
    pytest.main()