"""Independent geometric checks without loading a checkpoint."""
import numpy as np
import pytest
import torch
from suncompass.suncompass import SunCompass

MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]

@pytest.mark.parametrize('h,w,nh,nw',[(1080,1920,126,224),(1920,1080,224,126),(224,224,224,224),(101,224,101,224),(1,5000,1,224)])
def test_geometry_and_normalization(h,w,nh,nw):
    result=SunCompass().apply_letterbox_transforms(np.full((h,w,3),255,dtype=np.uint8))
    assert result.shape==(1,3,224,224) and result.dtype==torch.float32
    pixels=((result[0]*STD+MEAN)*255).round().byte().permute(1,2,0).numpy()
    expected=np.empty((224,224,3),dtype=np.uint8);expected[:]=[124,116,104]
    y,x=(224-nh)//2,(224-nw)//2
    expected[y:y+nh,x:x+nw]=255
    assert np.array_equal(pixels,expected)

def test_four_edges_reach_model_without_second_preprocessing():
    class Capture(torch.nn.Module):
        def forward(self,x,preprocessed=False):
            assert preprocessed
            self.seen=x.clone()
            return torch.tensor([[.5,-.5]])
    image=np.full((100,200,3),128,dtype=np.uint8)
    image[:10]=[255,0,0];image[-10:]=[0,255,0]
    image[:,:10]=[0,0,255];image[:,-10:]=[255,255,0]
    sun=SunCompass();sun._model=Capture();sun.device=torch.device('cpu')
    assert np.array_equal(sun.letterbox_predict(image),[.5,-.5])
    pixels=(sun.model.seen[0]*STD+MEAN).permute(1,2,0).numpy()
    for y,x,c in [(58,112,[1,0,0]),(165,112,[0,1,0]),(112,2,[0,0,1]),(112,222,[1,1,0])]:
        assert np.allclose(pixels[y,x],c,atol=1e-6)
