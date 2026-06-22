from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder_interface import EncoderInterface
from lhotse.dataset import SpecAugment
from .scaling import ScaledLinear

from utils.utils import add_sos, make_pad_mask, time_warp, torch_autocast


class AudioEmbeddingSelector(nn.Module):
    """Frame-wise soft selector over several audio embedding experts.

    Input/Output:
      x: (B, T, C)
      selected: (B, T, C)
      expert_out: (B, T, K, C), where K = num_selectors
    """

    def __init__(
        self,
        encoder_dim: int,
        num_selectors: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_selectors = num_selectors
        self.selector = nn.Linear(encoder_dim, num_selectors)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(encoder_dim),
                    nn.Linear(encoder_dim, encoder_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(encoder_dim, encoder_dim),
                )
                for _ in range(num_selectors)
            ]
        )
        self.out_norm = nn.LayerNorm(encoder_dim)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        selector_logits = self.selector(x)  # (B, T, K)
        selector_weights = torch.softmax(selector_logits, dim=-1)  # (B, T, K)

        expert_out = torch.stack([expert(x) for expert in self.experts], dim=2)
        selected = (expert_out * selector_weights.unsqueeze(-1)).sum(dim=2)
        selected = self.out_norm(x + selected)

        if padding_mask is not None:
            selected = selected.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            expert_out = expert_out.masked_fill(padding_mask[:, :, None, None], 0.0)

        return selected, selector_logits, selector_weights, expert_out


class TwoLanguageCrossAttention(nn.Module):
    """Cross attention between two selected audio-embedding streams.

    This module assumes num_selectors >= 2 and uses the first two expert streams
    as two language/speaker/state views. If you only need hard 2-way selection,
    set num_audio_selectors=2.
    """
    def __init__(
        self,
        encoder_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lang0_to_lang1 = nn.MultiheadAttention(
            encoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.lang1_to_lang0 = nn.MultiheadAttention(
            encoder_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.fuse = nn.Sequential(
            nn.Linear(encoder_dim * 2, encoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.norm = nn.LayerNorm(encoder_dim)

    def forward(
        self,
        embedding: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert embedding.size(2) >= 2, embedding.shape
        lang0 = embedding[:, :, 0, :]  # (B, T, C)
        lang1 = embedding[:, :, 1, :]  # (B, T, C)

        lang0_ctx, _ = self.lang0_to_lang1(
            query=lang0,
            key=lang1,
            value=lang1,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        lang1_ctx, _ = self.lang1_to_lang0(
            query=lang1,
            key=lang0,
            value=lang0,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        fused = self.fuse(torch.cat([lang0_ctx, lang1_ctx], dim=-1))
        fused = self.norm((lang0 + lang1) * 0.5 + fused)

        if padding_mask is not None:
            fused = fused.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return fused


class DialogueStateClassifier(nn.Module):
    """Chunk/utterance-level dialogue state classifier from encoder outputs."""

    def __init__(
        self,
        encoder_dim: int,
        num_states: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, encoder_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_states),
        )

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> torch.Tensor:
        # masked mean pooling: (B, T, C) -> (B, C)
        padding_mask = make_pad_mask(encoder_out_lens).to(encoder_out.device)
        valid_mask = (~padding_mask).unsqueeze(-1).to(encoder_out.dtype)
        pooled = (encoder_out * valid_mask).sum(dim=1)
        pooled = pooled / encoder_out_lens.clamp(min=1).unsqueeze(-1).to(encoder_out.dtype)
        return self.classifier(pooled)

class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    
class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)
    
class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)
    
class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (is_layer_norm and is_group_norm) == False, (
                "layer norm and group norm are exclusive"
            )

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    @staticmethod
    def _conv_out_length(
        input_length: torch.Tensor,
        kernel_size: int,
        stride: int,
        padding: int = 0,
        dilation: int = 1,
    ) -> torch.Tensor:
        return torch.div(
            input_length + 2 * padding - dilation * (kernel_size - 1) - 1,
            stride,
            rounding_mode="floor",
        ) + 1

    def forward(self, x, x_lens):
        x_lens = x_lens.to(torch.long)
        # BxT -> BxCxT
        x = x.unsqueeze(1)
        
        for conv_block in self.conv_layers:
            conv = conv_block[0]
            x = conv_block(x)
            x_lens = self._conv_out_length(
                x_lens,
                kernel_size=conv.kernel_size[0],
                stride=conv.stride[0],
                padding=conv.padding[0],
                dilation=conv.dilation[0],
            )
        
        return x, x_lens

class DualAsrModelWav(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        attention_decoder: Optional[nn.Module] = None,
        feature_dim: int = 512,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_attention_decoder: bool = False,
        use_dialogue_state_classifier: bool = True,
        num_dialogue_states: int = 6,
        num_audio_selectors: int = 2,
        selector_num_heads: int = 4,
        selector_dropout: float = 0.1,
        use_selected_encoder_for_asr: bool = True,
        feature_grad_mult: float = 0.1,
        feature_enc_layers: List[Tuple[int, int, int]] = None
    ):
        """A joint CTC & Transducer ASR model.

        - Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks (http://imagine.enpc.fr/~obozinsg/teaching/mva_gm/papers/ctc.pdf)
        - Sequence Transduction with Recurrent Neural Networks (https://arxiv.org/pdf/1211.3711.pdf)
        - Pruned RNN-T for fast, memory-efficient ASR training (https://arxiv.org/pdf/2206.13236.pdf)

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module. It converts
            an input of shape (N, T, idim) to an output of of shape
            (N, T', odim), where T' = (T-3)//2-2 = (T-7)//2.
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, encoder_dim) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, encoder_dim) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, decoder_dim).
            It should contain one attribute: `blank_id`.
            It is used when use_transducer is True.
          joiner:
            It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
            Its output shape is (N, T, U, vocab_size). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
            It is used when use_transducer is True.
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
          use_attention_decoder:
            Whether use attention-decoder head. Default: False.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True, but got use_transducer={use_transducer}, use_ctc={use_ctc}"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.feature_grad_mult = feature_grad_mult
        self.feature_enc_layers = feature_enc_layers

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
        )

        self.layer_norm = nn.LayerNorm(feature_dim)

        self.encoder_embed = encoder_embed
        self.encoder = encoder

        self.use_transducer = use_transducer
        if use_transducer:
            # Modules for Transducer head
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            # Modules for CTC head
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.attention_decoder = attention_decoder
        else:
            assert attention_decoder is None

        # Optional dialogue-state branch.  This branch first selects/fuses
        # audio embeddings and then predicts a chunk-level dialogue state.
        # If use_selected_encoder_for_asr=True, the fused embedding is also
        # fed to Transducer/CTC/AED heads.
        self.use_dialogue_state_classifier = use_dialogue_state_classifier
        self.use_selected_encoder_for_asr = use_selected_encoder_for_asr
        if use_dialogue_state_classifier:
            assert num_audio_selectors >= 2, num_audio_selectors
            self.audio_embedding_selector = AudioEmbeddingSelector(
                encoder_dim=encoder_dim,
                num_selectors=num_audio_selectors,
                dropout=selector_dropout,
            )
            self.language_cross_attention = TwoLanguageCrossAttention(
                encoder_dim=encoder_dim,
                num_heads=selector_num_heads,
                dropout=selector_dropout,
            )
            self.dialogue_fusion = nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim * 3, encoder_dim),
                nn.SiLU(),
                nn.Dropout(selector_dropout),
                nn.Linear(encoder_dim, encoder_dim),
            )
            self.dialogue_fusion_norm = nn.LayerNorm(encoder_dim)
            self.dialogue_state_classifier = DialogueStateClassifier(
                encoder_dim=encoder_dim,
                num_states=num_dialogue_states,
                dropout=selector_dropout,
            )

    def forward_features(self, source: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features, feature_lens = self.feature_extractor(source, x_lens)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features, feature_lens = self.feature_extractor(source, x_lens)
        return features, feature_lens

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
        # logging.info(f"Memory allocated at entry: {torch.cuda.memory_allocated() // 1000000}M")
        x, x_lens = self.encoder_embed(x, x_lens)
        # logging.info(f"Memory allocated after encoder_embed: {torch.cuda.memory_allocated() // 1000000}M")

        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask)

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          targets:
            Target Tensor of shape (sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC log-prob
        ctc_output = self.ctc_output(encoder_out)  # (N, T, C)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_cr_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute CTC loss with consistency regularization loss.
        Args:
          encoder_out:
            Encoder output, of shape (2 * N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (2 * N,).
          targets:
            Target Tensor of shape (2 * sum(target_lengths)). The targets are assumed
            to be un-padded and concatenated within 1 dimension.
        """
        # Compute CTC loss
        ctc_output = self.ctc_output(encoder_out)  # (2 * N, T, C)
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),  # (T, 2 * N, C)
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )

        # Compute consistency regularization loss
        batch_size = ctc_output.shape[0]
        assert batch_size % 2 == 0, batch_size
        # exchange: [x1, x2] -> [x2, x1]
        exchanged_targets = torch.roll(ctc_output.detach(), batch_size // 2, dims=0)
        cr_loss = nn.functional.kl_div(
            input=ctc_output,
            target=exchanged_targets,
            reduction="none",
            log_target=True,
        )  # (2 * N, T, C)
        length_mask = make_pad_mask(encoder_out_lens).unsqueeze(-1)
        cr_loss = cr_loss.masked_fill(length_mask, 0.0).sum()

        return ctc_loss, cr_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.
        Args:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
        """
        # Now for the decoder, i.e., the prediction network
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)

        # sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)

        # decoder_out: [B, S + 1, decoder_dim]
        decoder_out = self.decoder(sos_y_padded)

        # Note: y does not start with SOS
        # y_padded : [B, S]
        y_padded = y.pad(mode="constant", padding_value=0)

        y_padded = y_padded.to(torch.int64)
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        # if self.training and random.random() < 0.25:
        #    lm = penalize_abs_values_gt(lm, 100.0, 1.0e-04)
        # if self.training and random.random() < 0.25:
        #    am = penalize_abs_values_gt(am, 30.0, 1.0e-04)

        with torch_autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        # ranges : [B, T, prune_range]
        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        # am_pruned : [B, T, prune_range, encoder_dim]
        # lm_pruned : [B, T, prune_range, decoder_dim]
        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # logits : [B, T, prune_range, vocab_size]

        # project_input=False since we applied the decoder's input projections
        # prior to do_rnnt_pruning (this is an optimization for speed).
        logits = self.joiner(am_pruned, lm_pruned, project_input=False)

        with torch_autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        use_cr_ctc: bool = False,
        use_spec_aug: bool = False,
        spec_augment: Optional[SpecAugment] = None,
        supervision_segments: Optional[torch.Tensor] = None,
        time_warp_factor: Optional[int] = 80,
        dialogue_state_labels: Optional[torch.Tensor] = None,
        dialogue_state_loss_scale: float = 1.0,
        return_dialogue_state_outputs: bool = False,
    ):
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
          prune_range:
            The prune range for rnnt loss, it means how many symbols(context)
            we are considering for each frame to compute the loss.
          am_scale:
            The scale to smooth the loss with am (output of encoder network)
            part
          lm_scale:
            The scale to smooth the loss with lm (output of predictor network)
            part
          use_cr_ctc:
            Whether use consistency-regularized CTC.
          use_spec_aug:
            Whether apply spec-augment manually, used only if use_cr_ctc is True.
          spec_augment:
            The SpecAugment instance that returns time masks,
            used only if use_cr_ctc is True.
          supervision_segments:
            An int tensor of shape ``(S, 3)``. ``S`` is the number of
            supervision segments that exist in ``features``.
            Used only if use_cr_ctc is True.
          time_warp_factor:
            Parameter for the time warping; larger values mean more warping.
            Set to ``None``, or less than ``1``, to disable.
            Used only if use_cr_ctc is True.

        Returns:
          Return the transducer losses, CTC loss, AED loss,
          and consistency-regularization loss in form of
          (simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, cr_loss)

        Note:
           Regarding am_scale & lm_scale, it will make the loss-function one of
           the form:
              lm_scale * lm_probs + am_scale * am_probs +
              (1-lm_scale-am_scale) * combined_probs
        """
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0, (x.shape, x_lens.shape, y.dim0)

        device = x.device
        x, x_lens = self.forward_features(x, x_lens)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)

        if use_cr_ctc:
            assert self.use_ctc
            if use_spec_aug:
                assert spec_augment is not None and spec_augment.time_warp_factor < 1
                # Apply time warping before input duplicating
                assert supervision_segments is not None
                x = time_warp(
                    x,
                    time_warp_factor=time_warp_factor,
                    supervision_segments=supervision_segments,
                )
                # Independently apply frequency masking and time masking to the two copies
                x = spec_augment(x.repeat(2, 1, 1))
            else:
                x = x.repeat(2, 1, 1)
            x_lens = x_lens.repeat(2)
            y = k2.ragged.cat([y, y], axis=0)
            if dialogue_state_labels is not None:
                dialogue_state_labels = dialogue_state_labels.repeat(2)

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        dialogue_state_loss = torch.empty(0, device=device)
        dialogue_state_logits = None
        selector_logits = None
        selector_weights = None

        if self.use_dialogue_state_classifier:
            padding_mask = make_pad_mask(encoder_out_lens).to(encoder_out.device)

            selected_out, selector_logits, selector_weights, expert_out = (
                self.audio_embedding_selector(encoder_out, padding_mask=padding_mask)
            )
            cross_out = self.language_cross_attention(
                expert_out, padding_mask=padding_mask
            )

            fused_out = self.dialogue_fusion(
                torch.cat([encoder_out, selected_out, cross_out], dim=-1)
            )
            fused_out = self.dialogue_fusion_norm(encoder_out + fused_out)
            fused_out = fused_out.masked_fill(padding_mask.unsqueeze(-1), 0.0)

            dialogue_state_logits = self.dialogue_state_classifier(
                fused_out, encoder_out_lens
            )
            if dialogue_state_labels is not None:
                dialogue_state_loss = F.cross_entropy(
                    dialogue_state_logits,
                    dialogue_state_labels.to(device).long(),
                    reduction="sum",
                ) * dialogue_state_loss_scale

            if self.use_selected_encoder_for_asr:
                encoder_out = fused_out
        
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        if self.use_transducer:
            # Compute transducer loss
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
            if use_cr_ctc:
                simple_loss = simple_loss * 0.5
                pruned_loss = pruned_loss * 0.5
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        if self.use_ctc:
            # Compute CTC loss
            targets = y.values
            if not use_cr_ctc:
                ctc_loss = self.forward_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                cr_loss = torch.empty(0)
            else:
                ctc_loss, cr_loss = self.forward_cr_ctc(
                    encoder_out=encoder_out,
                    encoder_out_lens=encoder_out_lens,
                    targets=targets,
                    target_lengths=y_lens,
                )
                ctc_loss = ctc_loss * 0.5
                cr_loss = cr_loss * 0.5
        else:
            ctc_loss = torch.empty(0)
            cr_loss = torch.empty(0)

        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y.to(device),
                ys_lens=y_lens.to(device),
            )
            if use_cr_ctc:
                attention_decoder_loss = attention_decoder_loss * 0.5
        else:
            attention_decoder_loss = torch.empty(0)

        losses = (
            simple_loss,
            pruned_loss,
            ctc_loss,
            attention_decoder_loss,
            cr_loss,
        )

        if not self.use_dialogue_state_classifier:
            return losses

        if return_dialogue_state_outputs:
            return losses + (dialogue_state_loss,), {
                "dialogue_state_logits": dialogue_state_logits,
                "selector_logits": selector_logits,
                "selector_weights": selector_weights,
            }

        if dialogue_state_labels is not None:
            return losses + (dialogue_state_loss,)

        return losses

