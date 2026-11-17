"""Optional GPU checks against the upstream inference interfaces."""
import os
import numpy as np
import pytest

@pytest.mark.skipif(not os.environ.get('PILOT_TEST_WEIGHTS'),reason='set PILOT_TEST_WEIGHTS for CUDA contract tests')
def test_instrumented_predictor_preserves_both_upstream_methods(tmp_path):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from PIL import Image
    from suncompass import SunCompass
    from pilot.runner import Predictor
    weights=os.environ['PILOT_TEST_WEIGHTS']
    # Deliberately asymmetric RGB channels catch color-order changes; scan has four crops.
    image=np.zeros((340,340,3),dtype=np.uint8)
    image[:,:,0]=np.arange(340,dtype=np.uint16)[:,None] % 256
    image[:,:,1]=53
    image[:,:,2]=np.arange(340,dtype=np.uint16)[None,:] % 256
    path=tmp_path/'rgb.png';Image.fromarray(image).save(path)
    instrumented=Predictor(weights)
    original=SunCompass(weights_path=weights, preprocessing="legacy");original.set_eval(dropout=False)
    tensor=original.apply_transforms(image)
    assert tensor.shape==(1,3,224,224)
    for method in ('single','scan'):
        with torch.inference_mode():
            expected=(original.predict if method=='single' else original.scan_predict)(image)
        actual=instrumented.predict(path,method)
        np.testing.assert_array_equal(actual['raw_outputs'],expected)
        assert actual['crop_count']==(1 if method=='single' else 4)
