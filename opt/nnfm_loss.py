import torch
import torchvision
from icecream import ic


def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf


def argmin_cos_distance(a, b, center=False):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]
    """
    if center:
        a = a - a.mean(2, keepdims=True)
        b = b - b.mean(2, keepdims=True)

    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_best.append(z_best_batch)
    z_best = torch.cat(z_best, dim=-1)

    return z_best


def feat_replace(a, b):
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()

    z_new = []
    for i in range(n):
        z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new


def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()


def gram_matrix(feature_maps, center=False):
    """
    feature_maps: b, c, h, w
    gram_matrix: b, c, c
    """
    b, c, h, w = feature_maps.size()
    features = feature_maps.view(b, c, h * w)
    if center:
        features = features - features.mean(dim=-1, keepdims=True)
    G = torch.bmm(features, torch.transpose(features, 1, 2))
    return G


class NNFMLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[]):
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nnfm_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        for x in loss_names:
            assert x in ['nnfm_loss', 'content_loss', 'gram_loss']

        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a


        loss_dict = dict([(x, 0.) for x in loss_names])
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nnfm_loss" in loss_names:
                target_feats = feat_replace(x_feats, s_feats)
                loss_dict["nnfm_loss"] += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                loss_dict["gram_loss"] += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)

            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                loss_dict["content_loss"] += torch.mean((content_feats - x_feats) ** 2)

        return loss_dict


""" VGG-16 Structure
Input image is [-1, 3, 224, 224]
-------------------------------------------------------------------------------
        Layer (type)               Output Shape         Param #     Layer index
===============================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792     
              ReLU-2         [-1, 64, 224, 224]               0               1
            Conv2d-3         [-1, 64, 224, 224]          36,928     
              ReLU-4         [-1, 64, 224, 224]               0               3
         MaxPool2d-5         [-1, 64, 112, 112]               0     
            Conv2d-6        [-1, 128, 112, 112]          73,856     
              ReLU-7        [-1, 128, 112, 112]               0               6
            Conv2d-8        [-1, 128, 112, 112]         147,584     
              ReLU-9        [-1, 128, 112, 112]               0               8
        MaxPool2d-10          [-1, 128, 56, 56]               0     
           Conv2d-11          [-1, 256, 56, 56]         295,168     
             ReLU-12          [-1, 256, 56, 56]               0              11
           Conv2d-13          [-1, 256, 56, 56]         590,080     
             ReLU-14          [-1, 256, 56, 56]               0              13
           Conv2d-15          [-1, 256, 56, 56]         590,080     
             ReLU-16          [-1, 256, 56, 56]               0              15
        MaxPool2d-17          [-1, 256, 28, 28]               0     
           Conv2d-18          [-1, 512, 28, 28]       1,180,160     
             ReLU-19          [-1, 512, 28, 28]               0              18
           Conv2d-20          [-1, 512, 28, 28]       2,359,808     
             ReLU-21          [-1, 512, 28, 28]               0              20
           Conv2d-22          [-1, 512, 28, 28]       2,359,808     
             ReLU-23          [-1, 512, 28, 28]               0              22
        MaxPool2d-24          [-1, 512, 14, 14]               0     
           Conv2d-25          [-1, 512, 14, 14]       2,359,808     
             ReLU-26          [-1, 512, 14, 14]               0              25
           Conv2d-27          [-1, 512, 14, 14]       2,359,808     
             ReLU-28          [-1, 512, 14, 14]               0              27
           Conv2d-29          [-1, 512, 14, 14]       2,359,808    
             ReLU-30          [-1, 512, 14, 14]               0              29
        MaxPool2d-31            [-1, 512, 7, 7]               0    
===============================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 218.39
Params size (MB): 56.13
Estimated Total Size (MB): 275.10
----------------------------------------------------------------
"""


if __name__ == '__main__':
    device = torch.device('cuda:0')
    nnfm_loss_fn = NNFMLoss(device)
    fake_output = torch.rand(1, 3, 256, 256).to(device)
    fake_style = torch.rand(1, 3, 256, 256).to(device)
    fake_content = torch.rand(1, 3, 256, 256).to(device)

    loss = nnfm_loss_fn(outputs=fake_output, styles=fake_style, contents=fake_content, loss_names=["nnfm_loss", "content_loss", "gram_loss"])
    ic(loss)

    fake_image_set = torch.rand(10, 256, 256, 3).to(device)
    fake_style = torch.rand(256, 256, 3).to(device)
    fake_image_set_new, color_tf = match_colors_for_image_set(fake_image_set, fake_style)
    ic(fake_image_set_new.shape, color_tf.shape)
