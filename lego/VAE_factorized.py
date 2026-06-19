"""Factorized tabular VAE for binary LEGO-Xtal representations.

The decoder is explicitly sequential:
    global       = D_global(z)
    Si skeleton  = D_Si_skel(z, global_context)
    Si params    = D_Si_param(z, global_context, sampled_Si_skeleton)
    O skeleton   = D_O_skel(z, global_context, complete_Si_context)
    O params     = D_O_param(z, global_context, complete_Si_context, sampled_O_skeleton)

Each block owns a separate DataTransformer, allowing categorical skeleton tokens
and continuous coordinates to be reconstructed with the existing LEGO loss.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, ModuleList, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pyxtal.symmetry import Group

# v22: narrow/broad training-Gaussian mixture with smooth W reweighting and noisy R0 refinement.

from .data_transformer import DataTransformer
from .base import BaseSynthesizer, random_state


class MLP(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dim = input_dim
        layers = []
        for width in hidden_dims:
            layers.extend([Linear(dim, width), ReLU()])
            dim = width
        layers.append(Linear(dim, output_dim))
        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super().__init__()
        dim = data_dim
        layers = []
        for width in compress_dims:
            layers.extend([Linear(dim, width), ReLU()])
            dim = width
        self.body = Sequential(*layers)
        self.fc_mu = Linear(dim, embedding_dim)
        self.fc_logvar = Linear(dim, embedding_dim)

    def forward(self, x):
        h = self.body(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class ConditionalDecoder(Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, output_dim)
        self.sigma = Parameter(torch.ones(output_dim) * 0.1)

    def forward(self, x):
        return self.net(x), self.sigma


def _activate_transformed(
    logits,
    output_info_list,
    temperature=1.0,
    hard=False,
    categorical_masks=None,
):
    """Convert raw decoder output to transformed-space values.

    Continuous spans use tanh. Categorical spans use probabilities during
    training and optionally hard one-hot samples during generation.
    """
    st = 0
    outputs = []
    for column_info in output_info_list:
        for span_info in column_info:
            ed = st + span_info.dim
            span = logits[:, st:ed]
            if span_info.activation_fn != "softmax":
                outputs.append(torch.tanh(span))
            else:
                if categorical_masks is not None and (st, ed) in categorical_masks:
                    mask = categorical_masks[(st, ed)]
                    if mask.shape != span.shape:
                        raise ValueError(
                            f"Categorical mask shape {tuple(mask.shape)} does not "
                            f"match logits span {tuple(span.shape)}."
                        )
                    span = span.masked_fill(~mask, -torch.inf)
                scale = float(temperature) if temperature and temperature > 0 else 1.0
                probs = torch.softmax(span / scale, dim=-1)
                if hard:
                    index = torch.multinomial(probs, 1).squeeze(1)
                    probs = torch.nn.functional.one_hot(
                        index, num_classes=span_info.dim
                    ).float()
                outputs.append(probs)
            st = ed
    return torch.cat(outputs, dim=1)



def _get_discrete_span_and_categories(transformer, column_name):
    """Return transformed span and category values for one discrete column."""
    st = 0
    for info in transformer._column_transform_info_list:
        ed = st + info.output_dimensions
        if info.column_name == column_name:
            if info.column_type != "discrete":
                raise ValueError(f"Column {column_name!r} is not discrete.")
            categories = list(info.transform.dummies)
            if len(categories) != info.output_dimensions:
                raise RuntimeError(
                    f"Category count mismatch for {column_name!r}: "
                    f"{len(categories)} versus {info.output_dimensions}."
                )
            return st, ed, categories
        st = ed
    raise KeyError(f"Discrete column {column_name!r} not found.")


def _parse_wp_token(token):
    """Parse a padded Wyckoff-index token into integer indices."""
    try:
        return [int(value) for value in str(token).strip().split("|")]
    except ValueError as exc:
        raise ValueError(f"Malformed Wyckoff token: {token!r}") from exc


def _get_column_span(transformer, column_name):
    """Return the transformed-space span for one original column."""
    st = 0
    for info in transformer._column_transform_info_list:
        ed = st + info.output_dimensions
        if info.column_name == column_name:
            return st, ed
        st = ed
    raise KeyError(f"Column {column_name!r} not found in transformer.")


def _masked_selected_reconstruction_loss(
    recon_x,
    x,
    sigmas,
    output_info,
    factor,
    include_span,
    row_weight,
):
    """Selected-span reconstruction loss with one scalar weight per row."""
    st = 0
    per_row = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
    used = False
    for column_info in output_info:
        for span_info in column_info:
            ed = st + span_info.dim
            if include_span(st, ed):
                used = True
                if span_info.activation_fn != "softmax":
                    std = sigmas[st:ed]
                    residual = x[:, st:ed] - torch.tanh(recon_x[:, st:ed])
                    per_row = per_row + (
                        (residual ** 2) / (2 * std ** 2)
                    ).sum(dim=1)
                    per_row = per_row + torch.log(std).sum()
                else:
                    per_row = per_row + torch.nn.functional.cross_entropy(
                        recon_x[:, st:ed],
                        torch.argmax(x[:, st:ed], dim=-1),
                        reduction="none",
                    )
            st = ed
    if not used:
        return recon_x.sum() * 0.0
    denom = row_weight.sum().clamp_min(1.0)
    return (per_row * row_weight).sum() * factor / denom


def _block_reconstruction_loss(recon_x, x, sigmas, output_info, factor):
    st = 0
    terms = []
    for column_info in output_info:
        for span_info in column_info:
            ed = st + span_info.dim
            if span_info.activation_fn != "softmax":
                std = sigmas[st:ed]
                residual = x[:, st:ed] - torch.tanh(recon_x[:, st:ed])
                terms.append(((residual ** 2) / (2 * std ** 2)).sum())
                terms.append(torch.log(std).sum() * x.size(0))
            else:
                terms.append(
                    cross_entropy(
                        recon_x[:, st:ed],
                        torch.argmax(x[:, st:ed], dim=-1),
                        reduction="sum",
                    )
                )
            st = ed
    if st != recon_x.size(1):
        raise RuntimeError("Loss span layout does not match decoder output.")
    return sum(terms) * factor / x.size(0)




def _merge_skeleton_and_coordinates(skeleton_x, coordinate_x, span):
    """Use the selected skeleton span and coordinate output everywhere else."""
    st, ed = span
    output = coordinate_x.clone()
    output[:, st:ed] = skeleton_x[:, st:ed]
    return output


def _selected_reconstruction_loss(
    recon_x,
    x,
    sigmas,
    output_info,
    factor,
    include_span,
):
    """Reconstruction loss over selected transformed spans only."""
    st = 0
    terms = []
    for column_info in output_info:
        for span_info in column_info:
            ed = st + span_info.dim
            if include_span(st, ed):
                if span_info.activation_fn != "softmax":
                    std = sigmas[st:ed]
                    residual = x[:, st:ed] - torch.tanh(recon_x[:, st:ed])
                    terms.append(((residual ** 2) / (2 * std ** 2)).sum())
                    terms.append(torch.log(std).sum() * x.size(0))
                else:
                    terms.append(
                        cross_entropy(
                            recon_x[:, st:ed],
                            torch.argmax(x[:, st:ed], dim=-1),
                            reduction="sum",
                        )
                    )
            st = ed
    if not terms:
        return recon_x.sum() * 0.0
    return sum(terms) * factor / x.size(0)

class FactorizedVAE(BaseSynthesizer):
    """Factorized VAE with chemistry-aware Si Wyckoff feasibility masking."""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(512, 512),
        decompress_dims=(512, 512),
        context_dim=128,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2.0,
        global_loss_weight=1.0,
        si_loss_weight=1.0,
        o_loss_weight=1.0,
        kl_weight=1.0,
        kl_warmup_epochs=0,
        predicted_context_start=0.0,
        predicted_context_end=0.8,
        cross_loss_weight=0.1,
        cross_onset=2.0,
        cross_cutoff=2.4,
        cross_batch_size=16,
        cuda=True,
        verbose=False,
        folder="LEGO-FactorizedVAE",
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = tuple(compress_dims)
        self.decompress_dims = tuple(decompress_dims)
        self.context_dim = context_dim
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_factor = loss_factor
        self.global_loss_weight = global_loss_weight
        self.si_loss_weight = si_loss_weight
        self.o_loss_weight = o_loss_weight
        self.kl_weight = kl_weight
        self.kl_warmup_epochs = kl_warmup_epochs
        self.predicted_context_start = predicted_context_start
        self.predicted_context_end = predicted_context_end
        self.cross_loss_weight = float(cross_loss_weight)
        self.cross_onset = float(cross_onset)
        self.cross_cutoff = float(cross_cutoff)
        self.cross_batch_size = int(cross_batch_size)
        self.verbose = verbose
        self.root_folder = folder

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)

        self.model_folder = os.path.join(folder, "models")
        self.samples_folder = os.path.join(folder, "samples")
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.samples_folder, exist_ok=True)

    def _context_probability(self, epoch):
        if self.epochs <= 1:
            return self.predicted_context_end
        fraction = epoch / float(self.epochs - 1)
        return self.predicted_context_start + fraction * (
            self.predicted_context_end - self.predicted_context_start
        )

    @staticmethod
    def _mix_context(true_x, predicted_x, predicted_probability):
        if predicted_probability <= 0:
            return true_x
        if predicted_probability >= 1:
            return predicted_x
        choose_predicted = (
            torch.rand(true_x.size(0), 1, device=true_x.device)
            < predicted_probability
        )
        return torch.where(choose_predicted, predicted_x, true_x)

    def _build_models(self, global_dim, si_dim, o_dim):
        total_dim = global_dim + si_dim + o_dim
        self.encoder = Encoder(total_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.global_decoder = ConditionalDecoder(
            self.embedding_dim, self.decompress_dims, global_dim
        ).to(self._device)
        self.global_context_encoder = MLP(
            global_dim, (self.context_dim,), self.context_dim
        ).to(self._device)

        self.si_skeleton_decoder = ConditionalDecoder(
            self.embedding_dim + self.context_dim, self.decompress_dims, si_dim
        ).to(self._device)
        self.si_skeleton_context_encoder = MLP(
            si_dim, (self.context_dim,), self.context_dim
        ).to(self._device)
        # One decoder per independent Si site. Each site sees the global
        # crystallographic context, the complete Si Wyckoff skeleton, and the
        # already established partial Si block.
        self.si_coordinate_decoders = ModuleList([
            ConditionalDecoder(
                self.embedding_dim + 3 * self.context_dim,
                self.decompress_dims,
                si_dim,
            )
            for _ in range(self.n_si_sites)
        ]).to(self._device)
        self.si_context_encoder = MLP(
            si_dim, (self.context_dim,), self.context_dim
        ).to(self._device)

        self.o_skeleton_decoder = ConditionalDecoder(
            self.embedding_dim + 2 * self.context_dim,
            self.decompress_dims,
            o_dim,
        ).to(self._device)
        self.o_skeleton_context_encoder = MLP(
            o_dim, (self.context_dim,), self.context_dim
        ).to(self._device)
        self.o_coordinate_decoder = ConditionalDecoder(
            self.embedding_dim + 3 * self.context_dim,
            self.decompress_dims,
            o_dim,
        ).to(self._device)

    def _all_modules(self):
        return [
            self.encoder,
            self.global_decoder,
            self.global_context_encoder,
            self.si_skeleton_decoder,
            self.si_skeleton_context_encoder,
            self.si_coordinate_decoders,
            self.si_context_encoder,
            self.o_skeleton_decoder,
            self.o_skeleton_context_encoder,
            self.o_coordinate_decoder,
        ]

    def save(self, filepath):
        state = {name: getattr(self, name) for name in [
            "encoder", "global_decoder", "global_context_encoder",
            "si_skeleton_decoder", "si_skeleton_context_encoder",
            "si_coordinate_decoders", "si_context_encoder",
            "o_skeleton_decoder", "o_skeleton_context_encoder",
            "o_coordinate_decoder", "global_transformer", "si_transformer",
            "o_transformer",
        ]}
        state.update({
            "device": str(self._device),
            "config": {
                "embedding_dim": self.embedding_dim,
                "context_dim": self.context_dim,
                "decoder_stages": 5,
                "si_coordinate_mode": "sitewise_autoregressive",
                "si_skeleton_conditioning": "training_gaussian_mixture_gate_plus_smooth_packing_bias",
                "si_coordinate_conditioning": "best_packing_realization_from_selected_skeleton",
            },
        })
        joblib.dump(state, filepath)

    def load(self, filepath):
        state = joblib.load(filepath)
        for name, value in state.items():
            if name not in {"device", "config"}:
                setattr(self, name, value)
        self._device = torch.device(state.get("device", "cpu"))
        for module in self._all_modules():
            module.to(self._device)
        return self

    @random_state
    def fit(
        self,
        global_data,
        si_data,
        o_data,
        global_discrete_columns=(),
        si_discrete_columns=(),
        o_discrete_columns=(),
        geometry_data=None,
    ):
        if not (len(global_data) == len(si_data) == len(o_data)):
            raise ValueError("Global, Si, and O blocks must have equal row counts.")

        self.global_transformer = DataTransformer()
        self.si_transformer = DataTransformer()
        self.o_transformer = DataTransformer()
        self.global_transformer.fit(global_data, global_discrete_columns)
        self.si_transformer.fit(si_data, si_discrete_columns)
        self.o_transformer.fit(o_data, o_discrete_columns)

        global_x = self.global_transformer.transform(global_data).astype("float32")
        si_x = self.si_transformer.transform(si_data).astype("float32")
        o_x = self.o_transformer.transform(o_data).astype("float32")
        np.save(os.path.join(self.root_folder, "train_global.npy"), global_x)
        np.save(os.path.join(self.root_folder, "train_si.npy"), si_x)
        np.save(os.path.join(self.root_folder, "train_o.npy"), o_x)

        self.si_skeleton_span = _get_discrete_span_and_categories(
            self.si_transformer, "si_skeleton_token"
        )[:2]
        self.o_skeleton_span = _get_discrete_span_and_categories(
            self.o_transformer, "o_skeleton_token"
        )[:2]

        _, _, si_categories = _get_discrete_span_and_categories(
            self.si_transformer, "si_skeleton_token"
        )
        parsed_si_categories = [_parse_wp_token(token) for token in si_categories]
        self.n_si_sites = max(len(token) for token in parsed_si_categories)
        self.si_site_spans = [
            tuple(
                _get_column_span(self.si_transformer, f"si_{axis}{site}")
                for axis in ("u0_", "u1_", "u2_")
            )
            for site in range(self.n_si_sites)
        ]
        active_lookup = np.zeros(
            (len(parsed_si_categories), self.n_si_sites), dtype=np.float32
        )
        for category, token in enumerate(parsed_si_categories):
            for site, wp in enumerate(token[:self.n_si_sites]):
                active_lookup[category, site] = float(wp >= 0)
        self.si_active_lookup = torch.from_numpy(active_lookup).to(self._device)

        self._build_models(global_x.shape[1], si_x.shape[1], o_x.shape[1])
        row_ids = torch.arange(len(global_x), device=self._device, dtype=torch.long)
        dataset = TensorDataset(
            torch.from_numpy(global_x).to(self._device),
            torch.from_numpy(si_x).to(self._device),
            torch.from_numpy(o_x).to(self._device),
            row_ids,
        )
        self._geometry_data = geometry_data
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        parameters = [p for module in self._all_modules() for p in module.parameters()]
        optimizer = Adam(parameters, weight_decay=self.l2scale)
        scaler = torch.amp.GradScaler("cuda", enabled=self._device.type == "cuda")

        self.loss_values = []
        iterator = tqdm(range(self.epochs), disable=not self.verbose)
        for epoch in iterator:
            running = {
                "global": 0.0, "si_skeleton": 0.0, "si_coordinates": 0.0,
                "o_skeleton": 0.0, "o_coordinates": 0.0,
                "cross": 0.0, "teacher_cross": 0.0,
                "kl": 0.0,
            }
            count = 0
            predicted_probability = self._context_probability(epoch)
            current_kl_weight = self.kl_weight
            if self.kl_warmup_epochs > 0:
                current_kl_weight *= min(1.0, (epoch + 1) / float(self.kl_warmup_epochs))

            for global_b, si_b, o_b, row_id_b in loader:
                full_b = torch.cat([global_b, si_b, o_b], dim=1)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=self._device.type == "cuda"):
                    mu, std, logvar = self.encoder(full_b)
                    z = mu + torch.randn_like(std) * std

                    global_raw, global_sigma = self.global_decoder(z)
                    global_pred = _activate_transformed(global_raw, self.global_transformer.output_info_list)

                    # The normal autoregressive pathway may gradually consume
                    # predicted global context.
                    global_for_context = self._mix_context(
                        global_b, global_pred, predicted_probability
                    )
                    global_context = self.global_context_encoder(global_for_context)

                    # Stable teacher global context used specifically for
                    # learning P(R_Si | G, W_Si).
                    si_global_teacher_context = self.global_context_encoder(global_b)

                    si_skel_raw, si_skel_sigma = self.si_skeleton_decoder(
                        torch.cat([z, global_context], dim=1)
                    )
                    si_skel_pred_full = _activate_transformed(
                        si_skel_raw, self.si_transformer.output_info_list
                    )
                    si_true_skel = torch.zeros_like(si_b)
                    s0, s1 = self.si_skeleton_span
                    si_true_skel[:, s0:s1] = si_b[:, s0:s1]
                    si_pred_skel = torch.zeros_like(si_b)
                    si_pred_skel[:, s0:s1] = si_skel_pred_full[:, s0:s1]
                    # Normal mixed context is retained for downstream
                    # autoregressive conditioning.
                    si_skel_for_context = self._mix_context(
                        si_true_skel, si_pred_skel, predicted_probability
                    )
                    si_skel_context = self.si_skeleton_context_encoder(
                        si_skel_for_context
                    )

                    # Site-wise Si coordinate generation:
                    # P(R_i | G_teacher, W_Si,teacher, R_<i).
                    si_teacher_skel_context = self.si_skeleton_context_encoder(
                        si_true_skel
                    )
                    teacher_category = torch.argmax(
                        si_b[:, s0:s1], dim=1
                    )
                    si_active = self.si_active_lookup[teacher_category]

                    # Start from the teacher skeleton only. At each step, the
                    # already established teacher coordinates are exposed to
                    # the next site decoder (teacher forcing).
                    si_partial_teacher = si_true_skel.clone()
                    si_coord_pred = torch.zeros_like(si_b)
                    si_coordinate_loss = si_b.sum() * 0.0
                    si_coord_raw_for_geometry = torch.zeros_like(si_b)

                    for site, decoder in enumerate(self.si_coordinate_decoders):
                        partial_context = self.si_context_encoder(
                            si_partial_teacher
                        )
                        site_raw, site_sigma = decoder(
                            torch.cat(
                                [
                                    z,
                                    si_global_teacher_context,
                                    si_teacher_skel_context,
                                    partial_context,
                                ],
                                dim=1,
                            )
                        )
                        site_pred_full = _activate_transformed(
                            site_raw, self.si_transformer.output_info_list
                        )
                        site_spans = self.si_site_spans[site]
                        for st_site, ed_site in site_spans:
                            si_coord_pred[:, st_site:ed_site] = (
                                site_pred_full[:, st_site:ed_site]
                            )
                            si_coord_raw_for_geometry[:, st_site:ed_site] = (
                                site_raw[:, st_site:ed_site]
                            )

                        include = lambda st, ed, spans=site_spans: (
                            (st, ed) in spans
                        )
                        si_coordinate_loss = si_coordinate_loss + (
                            _masked_selected_reconstruction_loss(
                                site_raw,
                                si_b,
                                site_sigma,
                                self.si_transformer.output_info_list,
                                self.loss_factor,
                                include,
                                si_active[:, site],
                            )
                        )

                        # Teacher-force this site's true coordinates into the
                        # partial Si block for the next conditional step.
                        for st_site, ed_site in site_spans:
                            si_partial_teacher[:, st_site:ed_site] = (
                                si_b[:, st_site:ed_site]
                            )

                    si_pred = _merge_skeleton_and_coordinates(
                        si_skel_pred_full, si_coord_pred, self.si_skeleton_span
                    )
                    si_for_context = self._mix_context(si_b, si_pred, predicted_probability)
                    si_context = self.si_context_encoder(si_for_context)

                    o_skel_raw, o_skel_sigma = self.o_skeleton_decoder(
                        torch.cat([z, global_context, si_context], dim=1)
                    )
                    o_skel_pred_full = _activate_transformed(
                        o_skel_raw, self.o_transformer.output_info_list
                    )
                    o_true_skel = torch.zeros_like(o_b)
                    o0, o1 = self.o_skeleton_span
                    o_true_skel[:, o0:o1] = o_b[:, o0:o1]
                    o_pred_skel = torch.zeros_like(o_b)
                    o_pred_skel[:, o0:o1] = o_skel_pred_full[:, o0:o1]
                    o_skel_for_context = self._mix_context(
                        o_true_skel, o_pred_skel, predicted_probability
                    )
                    o_skel_context = self.o_skeleton_context_encoder(o_skel_for_context)

                    o_coord_raw, o_coord_sigma = self.o_coordinate_decoder(
                        torch.cat([z, global_context, si_context, o_skel_context], dim=1)
                    )

                    global_loss = _block_reconstruction_loss(
                        global_raw, global_b, global_sigma,
                        self.global_transformer.output_info_list, self.loss_factor,
                    )
                    si_skeleton_loss = _selected_reconstruction_loss(
                        si_skel_raw, si_b, si_skel_sigma,
                        self.si_transformer.output_info_list, self.loss_factor,
                        lambda st, ed: st == s0 and ed == s1,
                    )
                    o_skeleton_loss = _selected_reconstruction_loss(
                        o_skel_raw, o_b, o_skel_sigma,
                        self.o_transformer.output_info_list, self.loss_factor,
                        lambda st, ed: st == o0 and ed == o1,
                    )
                    o_coordinate_loss = _selected_reconstruction_loss(
                        o_coord_raw, o_b, o_coord_sigma,
                        self.o_transformer.output_info_list, self.loss_factor,
                        lambda st, ed: not (st == o0 and ed == o1),
                    )
                    kl_loss = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    ) / full_b.size(0)

                    # Optional cross-sublattice coordination regularizer.
                    cross_loss = si_coord_raw_for_geometry.sum() * 0.0
                    teacher_cross_loss = si_coord_raw_for_geometry.sum() * 0.0
                    if self.cross_loss_weight > 0 and self._geometry_data is not None:
                        n_cross = min(self.cross_batch_size, full_b.size(0))
                        cross_loss, teacher_cross_loss = self._geometry_data(
                            row_id_b[:n_cross],
                            global_raw[:n_cross],
                            si_coord_raw_for_geometry[:n_cross],
                            o_coord_raw[:n_cross],
                            self.global_transformer,
                            self.si_transformer,
                            self.o_transformer,
                            self.cross_onset,
                            self.cross_cutoff,
                            self._device,
                        )
                    si_loss = si_skeleton_loss + si_coordinate_loss
                    o_loss = o_skeleton_loss + o_coordinate_loss
                    loss = (
                        self.global_loss_weight * global_loss
                        + self.si_loss_weight * si_loss
                        + self.o_loss_weight * o_loss
                        + current_kl_weight * kl_loss
                        + self.cross_loss_weight * cross_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                for decoder in [
                    self.global_decoder, self.si_skeleton_decoder,
                    *list(self.si_coordinate_decoders),
                    self.o_skeleton_decoder, self.o_coordinate_decoder,
                ]:
                    decoder.sigma.data.clamp_(0.01, 1.0)

                bsz = full_b.size(0)
                count += bsz
                for key, value in [
                    ("global", global_loss), ("si_skeleton", si_skeleton_loss),
                    ("si_coordinates", si_coordinate_loss),
                    ("o_skeleton", o_skeleton_loss),
                    ("o_coordinates", o_coordinate_loss),
                    ("cross", cross_loss), ("teacher_cross", teacher_cross_loss),
                    ("kl", kl_loss),
                ]:
                    running[key] += value.detach().item() * bsz

            record = {"epoch": epoch + 1}
            for key, value in running.items():
                record[f"{key}_loss"] = value / count
            record["predicted_context_probability"] = predicted_probability
            record["kl_weight"] = current_kl_weight
            self.loss_values.append(record)
            total_display = (
                self.global_loss_weight * record["global_loss"]
                + self.si_loss_weight * (
                    record["si_skeleton_loss"] + record["si_coordinates_loss"]
                )
                + self.o_loss_weight * (
                    record["o_skeleton_loss"] + record["o_coordinates_loss"]
                )
                + current_kl_weight * record["kl_loss"]
                + self.cross_loss_weight * record["cross_loss"]
            )
            iterator.set_description(
                f"Loss {total_display:.3f} | coord {record['cross_loss']:.3f} "
                f"(w={self.cross_loss_weight:g}) | "
                f"Si sitewise R_i|G,W,R_<i | "
                f"ctx {predicted_probability:.2f}"
            )
            if (epoch + 1) % 25 == 0:
                self.save(os.path.join(
                    self.model_folder,
                    f"FactorizedVAE_checkpoint_epoch_{epoch + 1}.pkl",
                ))

        pd.DataFrame(self.loss_values).to_csv(
            os.path.join(self.root_folder, "factorized_loss.csv"), index=False
        )

    @staticmethod
    def _combined_sigma(skeleton_decoder, coordinate_decoder, span):
        sigma = coordinate_decoder.sigma.detach().clone()
        st, ed = span
        sigma[st:ed] = skeleton_decoder.sigma.detach()[st:ed]
        return sigma.cpu().numpy()

    def _combined_si_sigma(self, skeleton_span):
        sigma = torch.ones(
            self.si_transformer.output_dimensions,
            device=self._device,
        ) * 0.1
        for site, decoder in enumerate(self.si_coordinate_decoders):
            for st, ed in self.si_site_spans[site]:
                sigma[st:ed] = decoder.sigma.detach()[st:ed]
        st, ed = skeleton_span
        sigma[st:ed] = self.si_skeleton_decoder.sigma.detach()[st:ed]
        return sigma.cpu().numpy()

    @random_state
    def sample(
        self,
        samples,
        temperature=1.0,
        hard=True,
        enforce_sio2_multiplicity=True,
        max_independent_sites=None,
        si_skeleton_feasibility=None,
        si_feasibility_topk=16,
    ):
        """Sample G, mask Si skeletons by stoichiometry and packing feasibility, then generate coordinates."""
        for module in self._all_modules():
            module.eval()

        spg_st, spg_ed, spg_categories = _get_discrete_span_and_categories(
            self.global_transformer, "spg"
        )
        si_st, si_ed, si_categories = _get_discrete_span_and_categories(
            self.si_transformer, "si_skeleton_token"
        )
        o_st, o_ed, o_categories = _get_discrete_span_and_categories(
            self.o_transformer, "o_skeleton_token"
        )
        self.si_skeleton_span = (si_st, si_ed)
        self.o_skeleton_span = (o_st, o_ed)

        parsed_si = [_parse_wp_token(token) for token in si_categories]
        parsed_o = [_parse_wp_token(token) for token in o_categories]
        si_counts = [sum(wp >= 0 for wp in token) for token in parsed_si]
        o_counts = [sum(wp >= 0 for wp in token) for token in parsed_o]
        group_cache = {}
        if max_independent_sites is not None:
            max_independent_sites = int(max_independent_sites)
            if max_independent_sites <= 0:
                raise ValueError("max_independent_sites must be positive or None.")

        global_batches, global_realized_batches, si_batches, o_batches, valid_batches = [], [], [], [], []
        stats = {
            "rows": 0, "invalid_space_group": 0,
            "no_compatible_si_skeleton": 0, "invalid_si_skeleton": 0,
            "no_packing_feasible_si_skeleton": 0,
            "packing_conditioned_si_coordinates": 0,
            "no_compatible_o_skeleton": 0,
        }
        generated = 0
        with torch.no_grad():
            while generated < samples:
                current_batch = min(self.batch_size, samples - generated)
                z = torch.randn(current_batch, self.embedding_dim, device=self._device)

                global_raw, _ = self.global_decoder(z)
                global_x = _activate_transformed(
                    global_raw, self.global_transformer.output_info_list,
                    temperature=temperature, hard=hard,
                )
                global_context = self.global_context_encoder(global_x)

                # Realize the decoder Gaussian randomizer exactly once.  This
                # noisy cell is used both by the Si chemistry condition and by
                # the final output, preserving G diversity without a mismatch.
                global_batch_df = self.global_transformer.inverse_transform(
                    global_x.detach().cpu().numpy(),
                    self.global_decoder.sigma.detach().cpu().numpy(),
                )
                global_realized_batches.append(global_batch_df.copy())

                row_valid = torch.ones(current_batch, dtype=torch.bool, device=self._device)
                spg_ids = torch.argmax(global_x[:, spg_st:spg_ed], dim=1).cpu().tolist()
                row_multiplicities = [None] * current_batch
                for row, spg_id in enumerate(spg_ids):
                    try:
                        spg = int(round(float(spg_categories[spg_id])))
                    except (TypeError, ValueError, IndexError):
                        spg = -1
                    if not 1 <= spg <= 230:
                        row_valid[row] = False
                        stats["invalid_space_group"] += 1
                        continue
                    if spg not in group_cache:
                        try:
                            group = Group(spg)
                            group_cache[spg] = [
                                int(group[i].multiplicity) for i in range(len(group))
                            ]
                        except Exception:
                            row_valid[row] = False
                            stats["invalid_space_group"] += 1
                            continue
                    row_multiplicities[row] = group_cache[spg]

                # Stage 2: masked Si skeleton.
                si_skel_raw, _ = self.si_skeleton_decoder(
                    torch.cat([z, global_context], dim=1)
                )
                si_masks = None
                if enforce_sio2_multiplicity:
                    si_allowed = torch.zeros(
                        current_batch, len(si_categories), dtype=torch.bool,
                        device=self._device,
                    )
                    for row, mult in enumerate(row_multiplicities):
                        if mult is None:
                            si_allowed[row, 0] = True
                            continue
                        for si_id, si_token in enumerate(parsed_si):
                            si_wps = [wp for wp in si_token if wp >= 0]
                            if not si_wps or any(wp >= len(mult) for wp in si_wps):
                                continue
                            n_si = sum(mult[wp] for wp in si_wps)
                            for o_id, o_token in enumerate(parsed_o):
                                o_wps = [wp for wp in o_token if wp >= 0]
                                if not o_wps or any(wp >= len(mult) for wp in o_wps):
                                    continue
                                if max_independent_sites is not None and (
                                    si_counts[si_id] + o_counts[o_id] > max_independent_sites
                                ):
                                    continue
                                if sum(mult[wp] for wp in o_wps) == 2 * n_si:
                                    si_allowed[row, si_id] = True
                                    break
                        if not bool(si_allowed[row].any()):
                            row_valid[row] = False
                            stats["no_compatible_si_skeleton"] += 1
                            si_allowed[row, 0] = True
                    si_masks = {(si_st, si_ed): si_allowed}

                feasibility_best_free = None
                feasibility_best_score = None
                feasibility_logit_bias = None

                # Chemistry-aware feasibility filter:
                # keep only Si skeletons that can realize a broad nearest-neighbor
                # window inside the sampled cell after a short search over free
                # Wyckoff parameters.
                if si_skeleton_feasibility is not None:
                    if si_masks is None:
                        current_allowed = torch.ones(
                            current_batch,
                            len(si_categories),
                            dtype=torch.bool,
                            device=self._device,
                        )
                    else:
                        current_allowed = si_masks[(si_st, si_ed)].clone()

                    feasibility_result = si_skeleton_feasibility(
                        global_batch_df,
                        parsed_si,
                        current_allowed.detach().cpu().numpy(),
                        si_skel_raw[:, si_st:si_ed].detach().cpu().numpy(),
                        int(si_feasibility_topk),
                    )
                    if isinstance(feasibility_result, dict):
                        feasibility_allowed_np = feasibility_result["allowed"]
                        feasibility_best_free = feasibility_result.get("best_free")
                        feasibility_best_score = feasibility_result.get("best_score")
                        feasibility_logit_bias = feasibility_result.get("logit_bias")
                    else:
                        # Backward-compatible mask-only callback.
                        feasibility_allowed_np = feasibility_result
                        feasibility_best_free = None
                        feasibility_best_score = None
                        feasibility_logit_bias = None
                    feasibility_allowed = torch.as_tensor(
                        feasibility_allowed_np,
                        device=self._device,
                        dtype=torch.bool,
                    )
                    combined_allowed = current_allowed & feasibility_allowed

                    for row in range(current_batch):
                        if bool(row_valid[row]) and not bool(combined_allowed[row].any()):
                            row_valid[row] = False
                            stats["no_packing_feasible_si_skeleton"] += 1
                            # Keep one fallback category only to keep tensor sampling valid.
                            fallback = torch.argmax(
                                si_skel_raw[row, si_st:si_ed]
                            )
                            combined_allowed[row, fallback] = True

                    si_masks = {(si_st, si_ed): combined_allowed}

                    # Smooth chemistry conditioning: among candidates that pass
                    # the single stochastic Gaussian gate, retain decoder
                    # diversity but reweight logits by their packing energy.
                    if feasibility_logit_bias is not None:
                        bias = torch.as_tensor(
                            feasibility_logit_bias,
                            device=self._device,
                            dtype=si_skel_raw.dtype,
                        )
                        si_skel_raw[:, si_st:si_ed] = (
                            si_skel_raw[:, si_st:si_ed] + bias
                        )

                si_skel_full = _activate_transformed(
                    si_skel_raw, self.si_transformer.output_info_list,
                    temperature=temperature, hard=hard,
                    categorical_masks=si_masks,
                )
                si_skel_only = torch.zeros_like(si_skel_full)
                si_skel_only[:, si_st:si_ed] = si_skel_full[:, si_st:si_ed]
                si_skel_context = self.si_skeleton_context_encoder(si_skel_only)

                # Stage 3: site-wise Si coordinates conditioned on sampled
                # G, sampled W_Si, and previously generated Si sites.
                si_partial = si_skel_only.clone()
                si_coord_x = torch.zeros_like(si_skel_full)
                for site, decoder in enumerate(self.si_coordinate_decoders):
                    partial_context = self.si_context_encoder(si_partial)
                    site_raw, _ = decoder(
                        torch.cat(
                            [z, global_context, si_skel_context, partial_context],
                            dim=1,
                        )
                    )
                    site_x = _activate_transformed(
                        site_raw,
                        self.si_transformer.output_info_list,
                        temperature=temperature,
                        hard=hard,
                    )
                    for st_site, ed_site in self.si_site_spans[site]:
                        si_coord_x[:, st_site:ed_site] = (
                            site_x[:, st_site:ed_site]
                        )
                        si_partial[:, st_site:ed_site] = (
                            site_x[:, st_site:ed_site]
                        )
                # v20: realize the gradient-optimized chemistry-feasible R0 found while
                # selecting W0.  v18 discarded these coordinates and sampled an
                # unrelated R0, so feasibility of (G,W0) did not guarantee the
                # final Si sublattice.  Transform the selected free parameters
                # through the fitted Si transformer and overwrite only the
                # continuous coordinate spans.
                if feasibility_best_free is not None:
                    selected_si_ids = torch.argmax(
                        si_skel_full[:, si_st:si_ed], dim=1
                    ).detach().cpu().numpy()
                    override_records = []
                    use_override = np.zeros(current_batch, dtype=bool)
                    for row in range(current_batch):
                        category = int(selected_si_ids[row])
                        token = parsed_si[category]
                        record = {
                            "si_skeleton_token": si_categories[category]
                        }
                        free_values = feasibility_best_free[row, category]
                        for site in range(self.n_si_sites):
                            active = site < len(token) and int(token[site]) >= 0
                            if active and np.all(np.isfinite(free_values[site])):
                                values = free_values[site]
                                use_override[row] = True
                            else:
                                values = np.asarray(
                                    [-1.0, -1.0, -1.0] if not active
                                    else [0.0, 0.0, 0.0],
                                    dtype=float,
                                )
                            for axis in range(3):
                                record[f"si_u{axis}_{site}"] = float(values[axis])
                        override_records.append(record)

                    if bool(use_override.any()):
                        override_df = pd.DataFrame(override_records)
                        override_np = self.si_transformer.transform(
                            override_df
                        ).astype("float32")
                        override_x = torch.as_tensor(
                            override_np,
                            device=self._device,
                            dtype=si_coord_x.dtype,
                        )
                        override_mask = torch.as_tensor(
                            use_override,
                            device=self._device,
                            dtype=torch.bool,
                        )
                        stats["packing_conditioned_si_coordinates"] += int(
                            use_override.sum()
                        )
                        for site in range(self.n_si_sites):
                            for st_site, ed_site in self.si_site_spans[site]:
                                si_coord_x[override_mask, st_site:ed_site] = (
                                    override_x[override_mask, st_site:ed_site]
                                )

                si_x = _merge_skeleton_and_coordinates(
                    si_skel_full, si_coord_x, (si_st, si_ed)
                )
                si_context = self.si_context_encoder(si_x)

                # Stage 4: masked O skeleton conditioned on complete Si block.
                o_skel_raw, _ = self.o_skeleton_decoder(
                    torch.cat([z, global_context, si_context], dim=1)
                )
                o_masks = None
                if enforce_sio2_multiplicity:
                    si_ids = torch.argmax(si_x[:, si_st:si_ed], dim=1).cpu().tolist()
                    o_allowed = torch.zeros(
                        current_batch, len(o_categories), dtype=torch.bool,
                        device=self._device,
                    )
                    for row, (si_id, mult) in enumerate(zip(si_ids, row_multiplicities)):
                        if mult is None or not bool(row_valid[row]):
                            o_allowed[row, 0] = True
                            continue
                        si_wps = [wp for wp in parsed_si[si_id] if wp >= 0]
                        if not si_wps or any(wp >= len(mult) for wp in si_wps):
                            row_valid[row] = False
                            stats["invalid_si_skeleton"] += 1
                            o_allowed[row, 0] = True
                            continue
                        n_si = sum(mult[wp] for wp in si_wps)
                        for o_id, o_token in enumerate(parsed_o):
                            o_wps = [wp for wp in o_token if wp >= 0]
                            if not o_wps or any(wp >= len(mult) for wp in o_wps):
                                continue
                            if max_independent_sites is not None and (
                                si_counts[si_id] + o_counts[o_id] > max_independent_sites
                            ):
                                continue
                            if sum(mult[wp] for wp in o_wps) == 2 * n_si:
                                o_allowed[row, o_id] = True
                        if not bool(o_allowed[row].any()):
                            row_valid[row] = False
                            stats["no_compatible_o_skeleton"] += 1
                            o_allowed[row, 0] = True
                    o_masks = {(o_st, o_ed): o_allowed}
                o_skel_full = _activate_transformed(
                    o_skel_raw, self.o_transformer.output_info_list,
                    temperature=temperature, hard=hard,
                    categorical_masks=o_masks,
                )
                o_skel_only = torch.zeros_like(o_skel_full)
                o_skel_only[:, o_st:o_ed] = o_skel_full[:, o_st:o_ed]
                o_skel_context = self.o_skeleton_context_encoder(o_skel_only)

                # Stage 5: O coordinates conditioned on the sampled O skeleton.
                o_coord_raw, _ = self.o_coordinate_decoder(
                    torch.cat([z, global_context, si_context, o_skel_context], dim=1)
                )
                o_coord_x = _activate_transformed(
                    o_coord_raw, self.o_transformer.output_info_list,
                    temperature=temperature, hard=hard,
                )
                o_x = _merge_skeleton_and_coordinates(
                    o_skel_full, o_coord_x, (o_st, o_ed)
                )

                global_batches.append(global_x.cpu().numpy())
                si_batches.append(si_x.cpu().numpy())
                o_batches.append(o_x.cpu().numpy())
                valid_batches.append(row_valid.cpu().numpy())
                generated += current_batch

        global_array = np.concatenate(global_batches, axis=0)[:samples]
        si_array = np.concatenate(si_batches, axis=0)[:samples]
        o_array = np.concatenate(o_batches, axis=0)[:samples]
        valid_mask = np.concatenate(valid_batches, axis=0)[:samples]
        stats["rows"] = int(samples)

        # Use the exact noisy G realization already seen by the Si chemistry
        # conditioner.  No second inverse-transform draw is performed.
        global_df = pd.concat(
            global_realized_batches, ignore_index=True
        ).iloc[:samples].reset_index(drop=True)
        # The chemistry conditioner already returns a Gaussian-randomized,
        # smoothly refined R0.  Preserve it exactly during inverse transform;
        # adding a second independent perturbation would alter the conditioned
        # distribution.
        si_inverse_sigma = self._combined_si_sigma((si_st, si_ed)).copy()
        for site in range(self.n_si_sites):
            for st_site, ed_site in self.si_site_spans[site]:
                si_inverse_sigma[st_site:ed_site] = 0.0

        si_df = self.si_transformer.inverse_transform(
            si_array,
            si_inverse_sigma,
        )
        o_df = self.o_transformer.inverse_transform(
            o_array,
            self._combined_sigma(
                self.o_skeleton_decoder, self.o_coordinate_decoder,
                (o_st, o_ed),
            ),
        )
        return global_df, si_df, o_df, valid_mask, stats

    def set_device(self, device):
        self._device = torch.device(device)
        for module in self._all_modules():
            module.to(self._device)

