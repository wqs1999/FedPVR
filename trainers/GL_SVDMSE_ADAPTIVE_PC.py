"""GL_SVDMSE with Prototype Calibration and Adaptive Inference.

Prototype Calibration constructs class-wise visual prototypes from local
training image features and uses them to calibrate the prompt-induced text
classifier at inference time. Adaptive inference fuses personalized and
global-prompt logits using sample-wise reliability estimates.
"""

import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from Dassl.dassl.engine.trainer import TrainerX
from Dassl.dassl.metrics import compute_accuracy
from Dassl.dassl.utils import load_pretrained_weights, load_checkpoint
from Dassl.dassl.optim import build_optimizer, build_lr_scheduler
from Dassl.dassl.engine.build import TRAINER_REGISTRY

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": "GL_SVDMSE_ADAPTIVE",
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.GL_SVDMSE.N_CTX
        ctx_init = cfg.TRAINER.GL_SVDMSE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = cfg.TRAINER.GL_SVDMSE.N
        self.ratio = cfg.TRAINER.GL_SVDMSE.ratio
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            if cfg.TRAINER.GL_SVDMSE.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_global = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
                ctx_local = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)

            nn.init.normal_(ctx_global, std=0.02)
            nn.init.normal_(ctx_local, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx_global = nn.Parameter(ctx_global)
        self.ctx_local = nn.Parameter(ctx_local)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.repeat(self.N, 1)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.GL_SVDMSE.CLASS_TOKEN_POSITION

    def compute_null_space(self, global_ctx, ratio=0.8):
        global_ctx = global_ctx.view(-1, global_ctx.shape[-1])
        global_ctx = global_ctx.to(torch.float32)

        try:
            U, S, V = torch.svd(global_ctx)
        except RuntimeError as e:
            print(f"SVD failed on GPU: {e}")
            global_ctx_cpu = global_ctx.cpu()
            U, S, V = torch.svd(global_ctx_cpu)
            V = V.to(global_ctx.device)

        cutoff = int(S.shape[0] * (1 - ratio))
        V2 = V[:, cutoff:]

        return V2.to(global_ctx.dtype)

    def forward(self):
        ctx = self.ctx_local

        ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx = ctx.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx.shape[-1])

        ctx_global = self.ctx_global
        null_space = self.compute_null_space(ctx_global, self.ratio)

        ctx_global = ctx_global.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        ctx_global = ctx_global.permute(1, 0, 2, 3).contiguous().view(self.N * self.n_cls, self.n_ctx, ctx_global.shape[-1])

        ctx_flat = self.ctx_local.view(-1, self.ctx_local.shape[-1])
        null_space = null_space.to(ctx_flat.dtype)

        projected_ctx = torch.mm(ctx_flat, torch.mm(null_space, null_space.T))
        projected_ctx_local = projected_ctx.view(self.ctx_local.shape)
        projected_ctx_local = projected_ctx_local.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        projected_ctx_local = projected_ctx_local.permute(1, 0, 2, 3).contiguous().view(
            self.N * self.n_cls, self.n_ctx, ctx_global.shape[-1])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,
                    ctx,
                    suffix,
                ],
                dim=1,
            )
            prompts_global = torch.cat(
                [
                    prefix,
                    ctx_global,
                    suffix,
                ],
                dim=1,
            )
            prompts_projected_local = torch.cat(
                [
                    prefix,
                    projected_ctx_local,
                    suffix,
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, prompts_global, prompts_projected_local


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.n_cls = len(classnames)
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.N = cfg.TRAINER.GL_SVDMSE.N

        try:
            self.infer_mode = cfg.TRAINER.GL_SVDMSE.INFER_MODE
        except Exception:
            self.infer_mode = "local"

        if self.infer_mode not in ("local", "adaptive"):
            print(f"Warning: unknown infer_mode '{self.infer_mode}', falling back to 'local'")
            self.infer_mode = "local"

        # ================================================================
        #
        # }
        #
        # ================================================================
        self.feat_dim = clip_model.visual.output_dim
        self.prototype_dict = {}
        self.prototype_beta = cfg.TRAINER.GL_SVDMSE.PROTOTYPE_BETA

    # =====================================================================
    # =====================================================================

    def update_prototype(self, image_features, labels, idx=0):
        """Implementation note omitted from code release.

        Args:
        """
        with torch.no_grad():
            image_features = image_features.detach().float()

            if idx not in self.prototype_dict:
                device = image_features.device
                self.prototype_dict[idx] = {
                    "sum": torch.zeros(self.n_cls, self.feat_dim, device=device,
                                       dtype=torch.float32),
                    "cnt": torch.zeros(self.n_cls, device=device, dtype=torch.long),
                    "finalized": None,
                }

            proto = self.prototype_dict[idx]

            onehot = torch.zeros(labels.size(0), self.n_cls, device=labels.device,
                                 dtype=image_features.dtype)
            onehot.scatter_(1, labels.unsqueeze(1), 1.0)
            class_sum = onehot.T @ image_features
            class_cnt = onehot.sum(0)

            proto["sum"] += class_sum
            proto["cnt"] += class_cnt.long()
            proto["finalized"] = None

    def get_prototype(self, idx):
        """Implementation note omitted from code release."""
        if idx not in self.prototype_dict:
            return None

        proto = self.prototype_dict[idx]
        if proto["finalized"] is None:
            cnt = proto["cnt"].float().unsqueeze(1)

            if not hasattr(self, "_pc_printed_clients"):
                self._pc_printed_clients = set()
            if idx not in self._pc_printed_clients:
                nonzero = (cnt > 0).sum().item()
                total = cnt.size(0)
                total_samples = int(cnt.sum().item())
                print(f"[PrototypeCalib] client {idx}: {nonzero}/{total} classes "
                      f"have data, total samples accumulated = {total_samples}")
                self._pc_printed_clients.add(idx)

            safe_cnt = cnt.clamp(min=1.0)
            finalized = proto["sum"] / safe_cnt
            finalized = F.normalize(finalized, dim=-1)
            proto["finalized"] = finalized

        return proto["finalized"]

    # ---- Forward -------------------------------------------------------

    def forward(self, image, idx=None):
        tokenized_prompts = self.tokenized_prompts
        prompts, prompts_global, prompts_projected_local = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # ================================================================
        #   text_cal = (1-β) * text_local + β * visual_prototype
        # ================================================================
        if not self.training and self.prototype_beta > 0 and idx is not None:
            proto = self.get_prototype(idx)
            if proto is not None:
                proto = proto.to(dtype=text_features.dtype, device=text_features.device)
                if text_features.shape[0] == proto.shape[0]:
                    proto_for_text = proto
                elif (text_features.shape[0] == self.N * self.n_cls
                      and proto.shape[0] == self.n_cls):
                    proto_for_text = proto.unsqueeze(0).expand(self.N, -1, -1)
                    proto_for_text = proto_for_text.contiguous().view(self.N * self.n_cls, -1)
                else:
                    raise RuntimeError(
                        f"Prototype shape mismatch: text_features={text_features.shape}, "
                        f"proto={proto.shape}, N={self.N}, n_cls={self.n_cls}"
                    )
                text_features = (1 - self.prototype_beta) * text_features \
                                + self.prototype_beta * proto_for_text
                text_features = F.normalize(text_features, dim=-1)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features_global = self.text_encoder(prompts_global, tokenized_prompts)
        text_features_global = text_features_global / text_features_global.norm(dim=-1, keepdim=True)
        text_features_projected_local = self.text_encoder(prompts_projected_local, tokenized_prompts)
        text_features_projected_local = text_features_projected_local / text_features_projected_local.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        if self.training:
            logits = logit_scale * image_features @ text_features.t()
            logits_global = logit_scale * image_features @ text_features_global.t()
            return (logits, text_features_global, text_features,
                    text_features_projected_local, logits_global, image_features)


        local_logits = image_features @ text_features.t()

        # Global and projected-local classifiers are intentionally not calibrated
        # by prototypes. They are used for adaptive fusion and residual estimation.
        global_logits = image_features @ text_features_global.t()
        projected_logits = image_features @ text_features_projected_local.t()

        if self.infer_mode == "local":
            return logit_scale * local_logits

        local_logits_scaled = logit_scale * local_logits
        global_logits_scaled = logit_scale * global_logits
        projected_logits_scaled = logit_scale * projected_logits

        eps = 1e-6

        local_logits_f = local_logits_scaled.float()
        global_logits_f = global_logits_scaled.float()
        projected_logits_f = projected_logits_scaled.float()

        prob = F.softmax(local_logits_f, dim=-1)
        entropy = -(prob * torch.log(prob + eps)).sum(dim=-1)

        num_classes = local_logits_f.shape[-1]
        max_entropy = torch.log(torch.tensor(float(num_classes), device=local_logits_f.device))
        norm_entropy = entropy / (max_entropy + eps)
        norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)

        residual = torch.norm(local_logits_f - projected_logits_f, p=2, dim=-1) / (
            torch.norm(local_logits_f, p=2, dim=-1) + eps
        )

        res_min = residual.min()
        res_max = residual.max()
        norm_residual = (residual - res_min) / (res_max - res_min + eps)
        norm_residual = torch.clamp(norm_residual, 0.0, 1.0)

        alpha = torch.sqrt(norm_entropy * norm_residual + eps)
        alpha = torch.clamp(alpha, 0.0, 1.0)
        alpha = alpha.detach().unsqueeze(1).to(local_logits_scaled.dtype)

        # Fuse prototype-calibrated local logits with global-prompt logits:
        # z_i = (1 - alpha_i) * z_i^pc + alpha_i * z_i^g.
        final_logits = (1.0 - alpha) * local_logits_scaled + alpha * global_logits_scaled

        if not hasattr(self, "_adaptive_printed"):
            print(
                f"[Adaptive + PrototypeCalib] alpha mean={alpha.mean().item():.4f}, "
                f"min={alpha.min().item():.4f}, max={alpha.max().item():.4f}, "
                f"norm_entropy mean={norm_entropy.mean().item():.4f}, "
                f"norm_residual mean={norm_residual.mean().item():.4f}"
            )
            self._adaptive_printed = True

        return final_logits


@TRAINER_REGISTRY.register()
class GL_SVDMSE_ADAPTIVE_PC(TrainerX):
    """
    FedPHA + Adaptive Inference + Prototype Calibration.



    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.GL_SVDMSE.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.lambda_orthogonal = cfg.TRAINER.GL_SVDMSE.lambda_orthogonal
        self.alpha = cfg.TRAINER.GL_SVDMSE.alpha
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.GL_SVDMSE.PREC == "fp32" or cfg.TRAINER.GL_SVDMSE.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        if cfg.DATASET.NAME == "ImageNet":
            self.device = torch.device("cuda:0")
            device1 = torch.device("cuda")
            self.model.to(self.device)
            self.model.text_encoder.to(device1)
            self.model.text_encoder = nn.DataParallel(self.model.text_encoder)
        else:
            self.model.to(self.device)

        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GL_SVDMSE.PREC == "amp" else None

        print(f"[PrototypeCalib] beta={self.model.prototype_beta}, "
              f"feature_dim={self.model.feat_dim}, num_classes={self.model.n_cls}")

    def forward_backward(self, batch_idx, batch, idx=None, **kwargs):
        image, label = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.GL_SVDMSE.PREC

        if prec == "amp":
            with autocast():
                (output, global_features, local_features,
                 projected_local_features, output_global, image_features) = self.model(image)

                self.model.update_prototype(image_features.detach(), label, idx)

                pull_loss = F.mse_loss(local_features, projected_local_features)

                alpha = self.alpha
                push_loss = F.relu(alpha - torch.norm(local_features - global_features, dim=-1)).mean()
                lambda_pull = 1.0
                lambda_push = 1.0

                loss = F.cross_entropy(output, label)
                loss2 = F.cross_entropy(output_global, label)
                loss += loss2
                loss += lambda_pull * pull_loss + lambda_push * push_loss

            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            (output, global_features, local_features,
             projected_local_features, output_global, image_features) = self.model(image)

            self.model.update_prototype(image_features.detach(), label, idx)

            pull_loss = F.mse_loss(local_features, projected_local_features)

            alpha = self.alpha
            push_loss = F.relu(alpha - torch.norm(local_features - global_features, dim=-1)).mean()
            lambda_pull = 1.0
            lambda_push = 1.0

            loss = F.cross_entropy(output, label)
            loss2 = F.cross_entropy(output_global, label)
            loss += loss2
            loss += lambda_pull * pull_loss + lambda_push * push_loss

            self.model_backward_and_update(loss)

        with torch.no_grad():
            proto = self.model.prototype_dict.get(idx, None)
            if proto is not None:
                proto_nonzero_cls = (proto["cnt"] > 0).sum().item()
                proto_total_samples = proto["cnt"].sum().item()
            else:
                proto_nonzero_cls = 0
                proto_total_samples = 0

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "proto_nonzero_cls": proto_nonzero_cls,
            "proto_total_samples": proto_total_samples,
            "prototype_beta": self.model.prototype_beta,
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            self._models[name].load_state_dict(state_dict, strict=False)
