"""Independent pixel and model-boundary checks for Single Crop Max."""
import os
import numpy as np
import pytest
import torch
from PIL import Image
from suncompass import SunCompass

MEAN = torch.tensor([.485, .456, .406])[:, None, None]
STD = torch.tensor([.229, .224, .225])[:, None, None]

@pytest.mark.parametrize('h,w,y,x,side', [
    (1080, 1920, 0, 420, 1080), (1920, 1080, 420, 0, 1080),
    (225, 225, 0, 0, 225), (224, 227, 0, 1, 224),
    (227, 224, 1, 0, 224), (1, 6, 0, 2, 1), (6, 1, 2, 0, 1)])
def test_source_pixels_and_normalization(h, w, y, x, side):
    image = np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)
    tensor = SunCompass().apply_single_max_transforms(image)
    assert tensor.shape == (1, 3, 224, 224) and tensor.dtype == torch.float32
    pixels = ((tensor[0] * STD + MEAN) * 255).round().byte().permute(1, 2, 0).numpy()
    expected = np.array(Image.fromarray(image[y:y+side, x:x+side]).resize((224, 224), Image.Resampling.BILINEAR))
    np.testing.assert_array_equal(pixels, expected)


def test_retains_full_height_discards_sides_and_bypasses_internal_crop():
    class Capture(torch.nn.Module):
        calls = 0
        def forward(self, x, preprocessed=False):
            assert preprocessed
            self.calls += 1
            self.seen = x.clone()
            return torch.tensor([[.25, -.75]])
    image = np.full((224, 448, 3), 64, dtype=np.uint8)
    image[:, :112] = [0, 0, 255]; image[:, 336:] = [255, 255, 0]
    image[0, 112:336] = [255, 0, 0]; image[-1, 112:336] = [0, 255, 0]
    sun = SunCompass(); sun._model = Capture(); sun.device = torch.device('cpu')
    np.testing.assert_array_equal(sun.single_max_predict(image), [.25, -.75])
    assert sun.model.calls == 1
    pixels = ((sun.model.seen[0] * STD + MEAN) * 255).round().byte().permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(pixels, image[:, 112:336])


@pytest.mark.skipif(not os.environ.get('PILOT_TEST_WEIGHTS'), reason='set PILOT_TEST_WEIGHTS for CUDA contract')
def test_actual_resnet_boundary_and_predictor(tmp_path):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from pilot.runner import Predictor
    p = Predictor(os.environ['PILOT_TEST_WEIGHTS'])
    image = np.random.default_rng(5).integers(0, 256, (240, 430, 3), dtype=np.uint8)
    path = tmp_path/'image.png'; Image.fromarray(image).save(path)
    expected = p.sun.apply_single_max_transforms(image)
    seen = []
    hook = p.sun.model.resnet.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().cpu().clone()))
    wrapper = p.sun.apply_single_max_transforms
    for _ in range(2):
        result = p.predict(path, 'single-max')
        assert result['crop_count'] == 1 and result['transform_seconds'] > 0
        assert result['sun_deg'] is not None
        assert p.sun.apply_single_max_transforms is wrapper
    hook.remove()
    assert len(seen) == 2
    for actual in seen: torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('h,w', [(1080,1920),(227,224),(224,227),(1,6)])
def test_renderer_preview_is_unnormalized_network_input(h,w):
    import ast
    from pathlib import Path
    source=Path(__file__).resolve().parents[1]/'experiments/single_crop_max/render.py'
    tree=ast.parse(source.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='model_input')
    namespace={'Image':Image}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    image=np.random.default_rng(8).integers(0,256,(h,w,3),dtype=np.uint8)
    preview=np.array(namespace['model_input'](Image.fromarray(image),'single-max'))
    tensor=SunCompass().apply_single_max_transforms(image)
    pixels=((tensor[0]*STD+MEAN)*255).round().byte().permute(1,2,0).numpy()
    np.testing.assert_array_equal(preview,pixels)
