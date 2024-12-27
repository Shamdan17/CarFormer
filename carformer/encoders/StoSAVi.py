import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class StoSAVi(nn.Module):
    """SA model with stochastic kernel and additional prior_slots head.
    If loss_dict['kld_method'] = 'none', it becomes a standard SAVi model.
    """

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
            kernel_mlp=True,
        ),
        enc_dict=dict(
            num_channels=3,
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm="",
        ),
        dec_dict=dict(
            dec_channels=(128, 64, 64, 64, 64),
            dec_resolution=(8, 8),
            dec_ks=5,
            dec_norm="",
            upscale=False,
        ),
        pred_dict=dict(
            pred_type="transformer",
            pred_rnn=True,
            pred_norm_first=True,
            pred_num_layers=2,
            pred_num_heads=4,
            pred_ffn_dim=512,
            pred_sg_every=None,
        ),
        loss_dict=dict(
            use_post_recon_loss=True,
            recons_loss="mse",
            kld_method="var-0.01",  # 'none' to make it deterministic
        ),
        eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.clip_len = clip_len
        self.eps = eps

        # self.n_channels = n_channels

        self.slot_dict = slot_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.pred_dict = pred_dict
        self.loss_dict = loss_dict

        self.num_channels = 3
        self.use_ups = dec_dict.get("upscale", False)
        self.ups = torch.nn.Upsample(scale_factor=2, mode="bilinear")

        self._build_slot_attention()
        self._build_encoder()
        self._build_decoder()
        self._build_predictor()
        self._build_loss()

        # a hack for only extracting slots
        self.testing = False

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict["enc_out_channels"]
        self.num_slots = self.slot_dict["num_slots"]
        self.slot_size = self.slot_dict["slot_size"]
        self.slot_mlp_size = self.slot_dict["slot_mlp_size"]
        self.num_iterations = self.slot_dict["num_iterations"]

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size))
        )  # NOTE: slots are initialized with normal distribution

        # predict the (\mu, \sigma) to sample the `kernels` input to SA
        if self.slot_dict.get("kernel_mlp", True):
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2),
                nn.LayerNorm(self.slot_size * 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.slot_size * 2, self.slot_size * 2),
            )
        else:
            self.kernel_dist_layer = nn.Sequential(
                nn.Linear(self.slot_size, self.slot_size * 2),
            )

        # predict the `prior_slots`
        # useless, just for compatibility to load pre-trained weights
        self.prior_slot_layer = nn.Sequential(
            nn.Linear(self.slot_size, self.slot_size),
            nn.LayerNorm(self.slot_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.slot_size, self.slot_size),
        )

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_encoder(self):
        # Build Encoder
        # Conv CNN --> PosEnc --> MLP
        self.enc_channels = list(self.enc_dict["enc_channels"])  # CNN channels
        self.enc_ks = self.enc_dict["enc_ks"]  # kernel size in CNN
        self.enc_norm = self.enc_dict["enc_norm"]  # norm in CNN
        self.downsample = self.enc_dict.get("downsample", False)
        if self.downsample:
            self.visual_resolution = (96, 96)
        else:
            self.visual_resolution = (
                192,
                192,
            )  # #(64, 64)  # CNN out visual resolution
        self.visual_channels = self.enc_channels[-1]  # CNN out visual channels

        enc_layers = len(self.enc_channels) - 1
        self.encoder = nn.Sequential(
            *[
                conv_norm_act(
                    self.enc_channels[i],
                    self.enc_channels[i + 1],
                    kernel_size=self.enc_ks,
                    # 2x downsampling for 128x128 image
                    stride=(
                        2
                        if (i == 0 and self.resolution[0] == 192 and self.downsample)
                        else 1
                    ),  # 192) else 1,    # changed 128 -> 192
                    norm=self.enc_norm,
                    act="relu" if i != (enc_layers - 1) else "",
                )
                for i in range(enc_layers)
            ]
        )  # relu except for the last layer

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(
            self.visual_channels, self.visual_resolution
        )
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )

    def _build_decoder(self):
        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        self.dec_channels = self.dec_dict["dec_channels"]  # CNN channels
        self.dec_resolution = self.dec_dict["dec_resolution"]  # broadcast size
        self.dec_ks = self.dec_dict["dec_ks"]  # kernel size
        self.dec_norm = self.dec_dict["dec_norm"]  # norm in CNN
        assert self.dec_channels[0] == self.slot_size, "wrong in_channels for Decoder"
        modules = []
        in_size = self.dec_resolution[0]
        out_size = in_size
        stride = 2
        for i in range(len(self.dec_channels) - 1):
            if self.use_ups:
                if out_size * 2 == self.resolution[0]:
                    stride = 1
            else:
                if out_size == self.resolution[0]:
                    stride = 1

            if isinstance(self.dec_ks, list):
                cur_kernel_size = self.dec_ks[i]
            else:
                cur_kernel_size = self.dec_ks

            modules.append(
                deconv_norm_act(
                    self.dec_channels[i],
                    self.dec_channels[i + 1],
                    kernel_size=cur_kernel_size,
                    stride=stride,
                    norm=self.dec_norm,
                    act="relu",
                )
            )
            out_size = deconv_out_shape(
                out_size, stride, cur_kernel_size // 2, cur_kernel_size, stride - 1
            )

        if self.use_ups:
            out_size = out_size * 2

        assert_shape(
            self.resolution,
            (out_size, out_size),
            message="Output shape of decoder did not match input resolution. "
            "Try changing `decoder_resolution`.",
        )

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(
                self.dec_channels[-1],
                self.num_channels + 1,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )  # TODO: self.dec_dict['n_channels']
        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(
            self.slot_size, self.dec_resolution
        )

    def _build_predictor(self):
        """Predictor as in SAVi to transition slot from time t to t+1."""
        # Build Predictor
        pred_type = self.pred_dict.get("pred_type", "transformer")

        # Transformer (object interaction) --> LSTM (scene dynamic)
        if pred_type == "mlp":
            self.predictor = ResidualMLPPredictor(
                [self.slot_size, self.slot_size * 2, self.slot_size],
                norm_first=self.pred_dict["pred_norm_first"],
            )
        else:
            self.predictor = TransformerPredictor(
                self.slot_size,
                self.pred_dict["pred_num_layers"],
                self.pred_dict["pred_num_heads"],
                self.pred_dict["pred_ffn_dim"],
                norm_first=self.pred_dict["pred_norm_first"],
            )

        # wrap LSTM
        if self.pred_dict["pred_rnn"]:
            self.predictor = RNNPredictorWrapper(
                self.predictor,
                self.slot_size,
                self.slot_mlp_size,
                num_layers=1,
                rnn_cell="LSTM",  # NOTE: they are using lstm. change it to gru maybe?.
                sg_every=self.pred_dict["pred_sg_every"],
            )

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_post_recon_loss = self.loss_dict["use_post_recon_loss"]
        assert self.use_post_recon_loss
        # stochastic SAVi by sampling the kernels
        kld_method = self.loss_dict["kld_method"]
        # a smaller sigma for the prior distribution
        if "-" in kld_method:
            kld_method, kld_var = kld_method.split("-")
            self.kld_log_var = math.log(float(kld_var))
        else:
            self.kld_log_var = math.log(1.0)
        self.kld_method = kld_method
        assert self.kld_method in ["var", "none"]

    @property
    def device(self):
        return next(self.parameters()).device

    def _kld_loss(self, prior_dist, post_slots):
        """KLD between (mu, sigma) and (0 or mu, 1)."""
        raise NotImplementedError
        # if self.kld_method == "none":
        #     return torch.tensor(0.0).type_as(prior_dist)
        # assert prior_dist.shape[-1] == self.slot_size * 2
        # mu1 = prior_dist[..., : self.slot_size]
        # log_var1 = prior_dist[..., self.slot_size :]
        # mu2 = mu1.detach().clone()  # no penalty for mu
        # log_var2 = torch.ones_like(log_var1).detach() * self.kld_log_var
        # sigma1 = torch.exp(log_var1 * 0.5)
        # sigma2 = torch.exp(log_var2 * 0.5)
        # kld = (
        #     torch.log(sigma2 / sigma1)
        #     + (torch.exp(log_var1) + (mu1 - mu2) ** 2) / (2.0 * torch.exp(log_var2))
        #     - 0.5
        # )
        # return kld.sum(-1).mean()

    def _sample_dist(self, dist):
        """Sample values from Gaussian distribution."""
        assert dist.shape[-1] == self.slot_size * 2
        mu = dist[..., : self.slot_size]
        # not doing any stochastic
        if self.kld_method == "none":
            return mu
        log_var = dist[..., self.slot_size :]
        eps = torch.randn_like(mu).detach()
        sigma = torch.exp(log_var * 0.5)
        return mu + eps * sigma

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""

        encoder_out = self.encoder(img)
        encoder_out = self.encoder_pos_embedding(
            encoder_out
        )  # TODO: POSITIONAL EMBEDDINGS ?
        # `encoder_out` has shape: [B, C, H, W]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [B, C, H*W]
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        # chunk_size = 16
        # encoder_out_last = torch.zeros(
        #     encoder_out.shape[0],
        #     encoder_out.shape[1],
        #     self.enc_out_channels,
        #     dtype=encoder_out.dtype,
        #     device=encoder_out.device,
        # )
        encoder_out_last = self.encoder_out_layer(encoder_out)
        # for i in range(0, encoder_out.shape[0], chunk_size):
        #     encoder_out_last[i : i + chunk_size] = self.encoder_out_layer(
        #         encoder_out[i : i + chunk_size]
        #     )

        # encoder_out = torch.cat(chunks, dim=0)

        # encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]

        return encoder_out_last

    def encode(self, img, prev_slots=None, return_attn=False):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]

        # apply SlotAttn on video frames via reusing slots
        all_kernel_dist, all_post_slots = [], []
        if return_attn:
            attns = []
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = init_latents  # [B, N, C]
            else:
                latents = self.predictor(
                    prev_slots
                )  # [B, N, C]    # NOTE: rnn-based slot predictor.

            # stochastic `kernels` as SA input
            kernel_dist = self.kernel_dist_layer(latents)
            kernels = self._sample_dist(
                kernel_dist
            )  # if not stochastic just returns the mu value
            all_kernel_dist.append(kernel_dist)

            # perform SA to get `post_slots`
            if return_attn:
                post_slots, attn_scores = self.slot_attention(
                    encoder_out[:, idx], kernels, return_attn=return_attn
                )
                attns.append(attn_scores)
            else:
                post_slots = self.slot_attention(encoder_out[:, idx], kernels)
            all_post_slots.append(post_slots)

            # next timestep
            prev_slots = post_slots

        # (B, T, self.num_slots, self.slot_size)
        if return_attn:
            attns = torch.stack(
                [torch.stack([x for x in y], axis=1) for y in attns], axis=1
            )
        kernel_dist = torch.stack(all_kernel_dist, dim=1)
        post_slots = torch.stack(all_post_slots, dim=1)

        if return_attn:
            return kernel_dist, post_slots, encoder_out, attns
        else:
            return kernel_dist, post_slots, encoder_out, None

    def _reset_rnn(self):
        self.predictor.reset()

    def forward(self, img, return_attn=False):
        """A wrapper for model forward.

        If the input video is too long in testing, we manually cut it.
        """
        T = img.shape[1]
        # if T <= self.clip_len or self.training:
        return self._forward(img, None, return_attn=return_attn)

        # try to find the max len to input each time
        # clip_len = T
        # while True:
        #     try:
        #         _ = self._forward(img[:, :clip_len], None)
        #         del _
        #         torch.cuda.empty_cache()
        #         break
        #     except RuntimeError:  # CUDA out of memory
        #         import ipdb

        #         ipdb.set_trace()
        #         clip_len = clip_len // 2 + 1
        # # update `clip_len`
        # self.clip_len = max(self.clip_len, clip_len)
        # # no need to split
        # if clip_len == T:
        #     return self._forward(img, None)

        # # split along temporal dim
        # cat_dict = None
        # prev_slots = None
        # for clip_idx in range(0, T, clip_len):
        #     out_dict = self._forward(img[:, clip_idx : clip_idx + clip_len], prev_slots)
        #     # because this should be in test mode, we detach the outputs
        #     if cat_dict is None:
        #         cat_dict = {k: [v.detach()] for k, v in out_dict.items()}
        #     else:
        #         for k, v in out_dict.items():
        #             cat_dict[k].append(v.detach())
        #     prev_slots = cat_dict["post_slots"][-1][:, -1].detach().clone()
        #     del out_dict
        #     torch.cuda.empty_cache()
        # cat_dict = {k: torch_cat(v, dim=1) for k, v in cat_dict.items()}
        # return cat_dict

    def _forward(self, img, prev_slots=None, return_attn=False):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            prev_slots: [B, num_slots, slot_size] or None,
                the `post_slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        B, T = img.shape[:2]

        kernel_dist, post_slots, encoder_out, attns = self.encode(
            img, prev_slots=prev_slots, return_attn=return_attn
        )
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]

        """ post_recon_img, post_recons, post_masks, _ = \
                self.decode(post_slots.flatten(0, 1)) """
        out_dict = {
            #'post_recon_combined': post_recon_img, # TODO: added after
            "post_slots": post_slots,  # [B, T, num_slots, C]
            "kernel_dist": kernel_dist,  # [B, T, num_slots, 2C]
            "img": img,  # [B, T, 3, H, W]
        }

        if return_attn:
            out_dict["attns"] = attns

        if self.testing:
            return out_dict

        if self.use_post_recon_loss:
            post_recon_img, post_recons, post_masks, _ = self.decode(
                post_slots.flatten(0, 1)
            )

            post_dict = {
                "post_recon_combined": post_recon_img,  # [B*T, 3, H, W]
                "post_recons": post_recons,  # [B*T, num_slots, 3, H, W]
                "post_masks": post_masks,  # [B*T, num_slots, 1, H, W]
            }
            out_dict.update({k: v.unflatten(0, (B, T)) for k, v in post_dict.items()})

        return out_dict

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution
        num_channels = (
            self.num_channels
        )  # self.n_channels #8 #1 # self.enc_channels[0] # 8 #3

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(
            1, 1, self.dec_resolution[0], self.dec_resolution[1]
        )

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].
        if self.use_ups:
            height = height // 2
            width = width // 2

        out = out.view(bs, num_slots, self.num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]

        # NOTE: for each pixel, we decide on slots that has to gather their assigned energy to predict that little one pixel
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]

        # for each pixel you have num_slots probabilities. how do you measure if they diverge?

        # TODO: recons valus are very close I assume. we may want to sigmoid on it.
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]

        if self.use_ups:
            recon_combined = self.ups(recon_combined)
            BT, NS = recons.shape[:2]
            recons = self.ups(recons.flatten(0, 1)).unflatten(0, (BT, NS))
            masks = self.ups(masks.flatten(0, 1)).unflatten(0, (BT, NS))

        return recon_combined, recons, masks, slots


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots, return_attn=False):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)

        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        if return_attn:
            attns = []

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum("bnc,bmc->bnm", k, q)
            attn = F.softmax(attn_logits, dim=-1)

            # `attn` has shape: [B, num_inputs, num_slots].
            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum("bnm,bnc->bmc", attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            # Disable amp
            with torch.cuda.amp.autocast(enabled=False, dtype=torch.float16):
                # import ipdb

                # ipdb.set_trace()
                slots = self.gru(
                    updates.view(bs * self.num_slots, self.slot_size).float(),
                    slots_prev.view(bs * self.num_slots, self.slot_size).float(),
                )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

            if return_attn:
                attns.append(attn)

        if return_attn:
            return slots, attns

        return slots


def assert_shape(actual, expected, message=""):
    assert list(actual) == list(
        expected
    ), f"Expected shape: {expected} but passed shape: {actual}. {message}"


def build_grid(resolution):
    """return grid with shape [1, H, W, 4]."""
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges, indexing="ij")
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed(nn.Module):
    """Soft PE mapping normalized coords to feature maps."""

    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=hidden_size)
        self.register_buffer("grid", build_grid(resolution))  # [1, H, W, 4]

    def forward(self, inputs):
        """inputs: [B, C, H, W]."""
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2).contiguous()
        return inputs + emb_proj


def torch_cat(tensor_list, dim):
    if len(tensor_list[0].shape) <= dim:
        return torch.cat(tensor_list)
    return torch.cat(tensor_list, dim=dim)


def deconv_out_shape(
    in_size,
    stride,
    padding,
    kernel_size,
    out_padding,
    dilation=1,
):
    """Calculate the output shape of a ConvTranspose layer."""
    return (
        (in_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + out_padding
        + 1
    )


def conv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm="bn",
    act="relu",
    dim="2d",
):
    """Conv - Norm - Act."""
    conv = get_conv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ["bn", "in"],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(conv, normalizer, act_func)


def deconv_norm_act(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    norm="bn",
    act="relu",
    dim="2d",
):
    """ConvTranspose - Norm - Act."""
    deconv = get_deconv(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        dilation=dilation,
        groups=groups,
        bias=norm not in ["bn", "in"],
        dim=dim,
    )
    normalizer = get_normalizer(norm, out_channels, dim=dim)
    act_func = get_act_func(act)
    return nn.Sequential(deconv, normalizer, act_func)


def get_deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim="2d",
):
    """Get Conv layer."""
    return eval(f"nn.ConvTranspose{dim}")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        output_padding=stride - 1,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    dim="2d",
):
    """Get Conv layer."""
    return eval(f"nn.Conv{dim}")(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_normalizer(norm, channels, groups=16, dim="2d"):
    """Get normalization layer."""
    if norm == "":
        return nn.Identity()
    elif norm == "bn":
        return eval(f"nn.BatchNorm{dim}")(channels)
    elif norm == "gn":
        # 16 is taken from Table 3 of the GN paper
        return nn.GroupNorm(groups, channels)
    elif norm == "in":
        return eval(f"nn.InstanceNorm{dim}")(channels)
    elif norm == "ln":
        return nn.LayerNorm(channels)
    else:
        raise ValueError(f"Normalizer {norm} not supported!")


def get_act_func(act):
    """Get activation function."""
    if act == "":
        return nn.Identity()
    if act == "relu":
        return nn.ReLU(inplace=True)
    elif act == "leakyrelu":
        return nn.LeakyReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "swish":
        return nn.SiLU()
    elif act == "elu":
        return nn.ELU()
    elif act == "softplus":
        return nn.Softplus()
    elif act == "mish":
        return nn.Mish()
    elif act == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Activation function {act} not supported!")


"""Transition function used in SAVi and STEVE."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass


class TransformerPredictor(Predictor):
    """Transformer encoder."""

    def __init__(
        self,
        d_model=128,
        num_layers=1,
        num_heads=4,
        ffn_dim=256,
        norm_first=True,
    ):
        super().__init__()

        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_enc_layer, num_layers=num_layers
        )

    def forward(self, x):
        out = self.transformer_encoder(x)
        return out


class ResidualMLPPredictor(Predictor):
    """LN + residual MLP."""

    def __init__(self, channels, norm_first=True):
        super().__init__()

        assert len(channels) >= 2
        # since there is LN at the beginning of slot-attn
        # so only use a pre-ln here
        self.ln = nn.LayerNorm(channels[0])
        modules = []
        for i in range(len(channels) - 2):
            modules += [nn.Linear(channels[i], channels[i + 1]), nn.ReLU(inplace=True)]
        modules.append(nn.Linear(channels[-2], channels[-1]))
        self.mlp = nn.Sequential(*modules)

        self.norm_first = norm_first

    def forward(self, x):
        if not self.norm_first:
            res = x
        x = self.ln(x)
        if self.norm_first:
            res = x
        out = self.mlp(x)
        out = out + res
        return out


class RNNPredictorWrapper(Predictor):
    """Predictor wrapped in a RNN for sequential modeling."""

    def __init__(
        self,
        base_predictor,
        input_size=128,
        hidden_size=256,
        num_layers=1,
        rnn_cell="LSTM",
        sg_every=None,
    ):
        super().__init__()

        assert rnn_cell in ["LSTM", "GRU", "RNN"]
        self.base_predictor = base_predictor
        self.rnn = eval(
            f"nn.{rnn_cell.upper()}(input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers})"
        )
        self.step = 0
        self.hidden_state = None
        self.out_projector = nn.Linear(hidden_size, input_size)
        self.sg_every = sg_every  # detach all inputs every certain steps
        # in ICCV'21 PARTS (https://openaccess.thecvf.com/content/ICCV2021/papers/Zoran_PARTS_Unsupervised_Segmentation_With_Slots_Attention_and_Independence_Maximization_ICCV_2021_paper.pdf)
        # they detach RNN states every 4 steps to avoid overfitting
        # but we don't observe much difference in our experiments

    def forward(self, x):
        if self.sg_every is not None:
            if self.step % self.sg_every == 0 and self.step > 0:
                x = x.detach()
                # LSTM hiddens state is (h, c) tuple
                if not isinstance(self.hidden_state, torch.Tensor):
                    self.hidden_state = tuple([h.detach() for h in self.hidden_state])
                else:
                    self.hidden_state = self.hidden_state.detach()
        # `x` should have shape of [B, ..., C]
        out = self.base_predictor(x)
        out_shape = out.shape
        self.rnn.flatten_parameters()
        out, self.hidden_state = self.rnn(
            out.view(1, -1, out_shape[-1]), self.hidden_state
        )
        out = self.out_projector(out[0]).view(out_shape)
        self.step += 1
        return out

    def burnin(self, x):
        """Warm up the RNN by first few steps inputs."""
        self.reset()
        # `x` should have shape of [B, T, ..., C]
        B, T = x.shape[:2]
        out = self.base_predictor(x.flatten(0, 1)).unflatten(0, (B, T))
        out = out.transpose(1, 0).reshape(T, -1, x.shape[-1])
        _, self.hidden_state = self.rnn(out, self.hidden_state)
        self.step = T

    def reset(self):
        """Clear the RNN hidden state."""
        self.step = 0
        self.hidden_state = None
