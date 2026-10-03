"""Rebuild rife426.onnx and rife426_fast.onnx from the official RIFE v4.26 weights.

    python tools/export_rife.py        # needs torch and onnx; writes the two .onnx files to the repo root

Downloads RIFEv4.26_0921.zip (hzwer/RIFE on Hugging Face, MIT), replaces the warp with a resolution-agnostic
one so one graph serves any frame size, and exports:
  rife426.onnx       full-quality, input padded to a multiple of 64
  rife426_fast.onnx  RIFE's scale=0.5 mode (motion at half resolution, full-res warp), padded to a multiple of 128
Inputs: img0, img1 float32 [1,3,H,W] in 0..1, t float32 [1,1,1,1]. Output: out [1,3,H,W].
"""
import io, os, sys, types, urllib.request, zipfile
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(ROOT, "tools", ".rife")
URL = "https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip"


def warp(x, flow):
    H, W = x.shape[2], x.shape[3]
    gx = torch.arange(W, dtype=flow.dtype).view(1, 1, 1, -1)
    gy = torch.arange(H, dtype=flow.dtype).view(1, 1, -1, 1)
    gxs = (gx + flow[:, 0:1]) * (2.0 / (W - 1)) - 1.0
    gys = (gy + flow[:, 1:2]) * (2.0 / (H - 1)) - 1.0
    grid = torch.cat((gxs, gys), 1).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def load_ifnet():
    src_dir = os.path.join(WORK, "RIFEv4.26_0921")
    if not os.path.isdir(src_dir):
        os.makedirs(WORK, exist_ok=True)
        print("downloading", URL)
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())).extractall(WORK)
    code = open(os.path.join(src_dir, "IFNet_HDv3.py")).read()
    code = code.replace("from model.warplayer import warp", "")
    # accept a [1,1,1,1] timestep tensor and broadcast it, instead of repeat() by the image size
    code = code.replace("timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])",
                        "timestep = timestep * torch.ones_like(img0[:, :1])")
    mod = types.ModuleType("IFNet_HDv3"); mod.warp = warp
    exec(compile(code, "IFNet_HDv3.py", "exec"), mod.__dict__)
    net = mod.IFNet()
    sd = torch.load(os.path.join(src_dir, "flownet.pkl"), map_location="cpu")
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items() if "module." in k}, strict=False)
    return net.eval()


class Full(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, img0, img1, t):
        _, _, merged = s.n(torch.cat((img0, img1), 1), t, [16, 8, 4, 2, 1])
        return merged[4].clamp(0, 1)


class Fast(torch.nn.Module):
    def __init__(s, n): super().__init__(); s.n = n
    def forward(s, img0, img1, t):
        a = F.interpolate(img0, scale_factor=0.5, mode="bilinear", align_corners=False)
        b = F.interpolate(img1, scale_factor=0.5, mode="bilinear", align_corners=False)
        flows, mask, _ = s.n(torch.cat((a, b), 1), t, [16, 8, 4, 2, 1])
        flow = F.interpolate(flows[4], scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        m = torch.sigmoid(F.interpolate(mask, scale_factor=2.0, mode="bilinear", align_corners=False))
        return (warp(img0, flow[:, :2]) * m + warp(img1, flow[:, 2:4]) * (1 - m)).clamp(0, 1)


if __name__ == "__main__":
    net = load_ifnet()
    for name, model, (h, w) in [("rife426.onnx", Full(net), (256, 448)), ("rife426_fast.onnx", Fast(net), (256, 512))]:
        a, b, t = torch.rand(1, 3, h, w), torch.rand(1, 3, h, w), torch.full((1, 1, 1, 1), 0.5)
        with torch.no_grad():
            torch.onnx.export(model.eval(), (a, b, t), os.path.join(ROOT, name), input_names=["img0", "img1", "t"],
                              output_names=["out"], opset_version=17, dynamo=False, do_constant_folding=True,
                              dynamic_axes={k: {2: "H", 3: "W"} for k in ("img0", "img1", "out")})
        print("wrote", name)
