import pandas as pd
import pytest
from model import GLOBAL_REGIONS,generate_synthetic_dataset,prepare_features

def test_data_reproducibility():
    pd.testing.assert_frame_equal(generate_synthetic_dataset(100),generate_synthetic_dataset(100))

def test_dataset_schema_and_ranges():
    df=generate_synthetic_dataset(250)
    assert list(df.columns)==['Area','Bedrooms','Age','Region','Price']
    assert df.Area.between(700,5999).all()
    assert df.Bedrooms.between(1,7).all()
    assert df.Age.between(0,69).all()
    assert df.Region.isin(GLOBAL_REGIONS).all()
    assert (df.Price>0).all()

def test_feature_shape_and_region_encoding():
    x=prepare_features(1800,3,8,'Asia')
    assert x.shape==(1,3+len(GLOBAL_REGIONS))
    assert x['Region_Asia'].iloc[0]==1

def test_invalid_region():
    with pytest.raises(ValueError): prepare_features(1800,3,8,'Unknown')
