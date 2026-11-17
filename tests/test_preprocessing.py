import os
import numpy as np
from PIL import Image
import pytest
import torch

from suncompass import SunCompass
from suncompass.data_loading import get_transforms
from suncompass.preprocessing import Letterbox, MEAN, STD


@pytest.mark.parametrize("height,width,content_height,content_width", [
    (512,1024,512,1024), (1024,512,1024,512), (220,320,704,1024),
    (1,5000,1,1024), (1024,1024,1024,1024), (101,224,462,1024)])
def test_1024_geometry_and_train_inference_parity(height,width,content_height,content_width):
    image = np.full((height,width,3),255,dtype=np.uint8)
    train, evaluation = get_transforms(resize_size=1024, preprocessing="letterbox", augment=False)
    tensor = SunCompass(preprocessing="letterbox").apply_transforms(image)[0]
    torch.testing.assert_close(train(Image.fromarray(image)), tensor, rtol=0, atol=0)
    torch.testing.assert_close(evaluation(Image.fromarray(image)), tensor, rtol=0, atol=0)
    torch.testing.assert_close(SunCompass(preprocessing="letterbox").apply_letterbox_transforms(image)[0], tensor, rtol=0, atol=0)
    pixels = ((tensor*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None])*255).round().byte().permute(1,2,0).numpy()
    expected=np.empty((1024,1024,3),dtype=np.uint8);expected[:]=[124,116,104]
    y,x=(1024-content_height)//2,(1024-content_width)//2
    expected[y:y+content_height,x:x+content_width]=255
    np.testing.assert_array_equal(pixels, expected)


def test_letterbox_predict_keeps_all_edges_and_bypasses_second_preprocessing():
    class Capture(torch.nn.Module):
        def forward(self,x,preprocessed=False):
            assert preprocessed and x.shape == (1,3,1024,1024)
            self.seen=x.clone()
            return torch.tensor([[.5,-.25]])
    image=np.full((512,1024,3),128,dtype=np.uint8)
    image[:10]=[255,0,0];image[-10:]=[0,255,0]
    image[:,:10]=[0,0,255];image[:,-10:]=[255,255,0]
    sun=SunCompass(preprocessing="letterbox");sun._model=Capture();sun.device=torch.device("cpu")
    np.testing.assert_array_equal(sun.predict(image), [.5,-.25])
    pixels=(sun.model.seen[0]*torch.tensor(STD)[:,None,None]+torch.tensor(MEAN)[:,None,None]).permute(1,2,0)
    for y,x,color in [(258,512,[1,0,0]),(765,512,[0,1,0]),(512,2,[0,0,1]),(512,1022,[1,1,0])]:
        torch.testing.assert_close(pixels[y,x],torch.tensor(color,dtype=torch.float32),atol=1e-6,rtol=0)


def test_normalized_raw_tensor_is_rejected():
    with pytest.raises(ValueError, match="preprocessed=True"):
        Letterbox()(torch.full((3,20,30), -1.0))


@pytest.mark.skipif(not os.environ.get("PILOT_TEST_WEIGHTS"), reason="CUDA model contract needs weights")
def test_real_resnet_default_1024_boundary():
    sun=SunCompass(weights_path=os.environ["PILOT_TEST_WEIGHTS"], preprocessing="letterbox");sun.set_eval()
    seen=[]
    hook=sun.model.resnet.conv1.register_forward_pre_hook(lambda _,args:seen.append(tuple(args[0].shape)))
    image=np.full((512,1024,3),128,dtype=np.uint8)
    with torch.inference_mode(): output=sun.predict(image)
    hook.remove()
    assert seen==[(1,3,1024,1024)] and output.shape==(2,) and np.isfinite(output).all()


def test_612_letterbox_matches_training_and_inference():
    image = np.full((576, 1024, 3), 255, dtype=np.uint8)
    train, evaluation = get_transforms(resize_size=612, preprocessing="letterbox", augment=False)
    prediction_input = SunCompass(image_size=612, preprocessing="letterbox").apply_transforms(image)[0]
    assert prediction_input.shape == (3, 612, 612)
    torch.testing.assert_close(train(Image.fromarray(image)), prediction_input)
    torch.testing.assert_close(evaluation(Image.fromarray(image)), prediction_input)
