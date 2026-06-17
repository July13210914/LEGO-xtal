"""Factorized tabular VAE for binary LEGO-Xtal representations.

The decoder is explicitly sequential:
    global = D_global(z)
    Si     = D_Si(z, global_context)
    O      = D_O(z, global_context, Si_context)

Each block owns a separate DataTransformer, allowing categorical skeleton tokens
and continuous coordinates to be reconstructed with the existing LEGO loss.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


def _activate_transformed(logits, output_info_list, temperature=1.0, hard=False):
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


class FactorizedVAE(BaseSynthesizer):
    """Shared encoder with global, Si, and O conditional decoder branches."""

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
        self.verbose = verbose
        self.root_folder = folder

        if not cuda or not torch.cuda.is_available():
            device = "cpu"
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"
        self._device = torch.device(device)
        print(f"FactorizedVAE device: {self._device}")

        if self._device.type == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name(self._device)}")

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
        self.encoder = Encoder(
            total_dim, self.compress_dims, self.embedding_dim
        ).to(self._device)
        self.global_decoder = ConditionalDecoder(
            self.embedding_dim, self.decompress_dims, global_dim
        ).to(self._device)
        self.global_context_encoder = MLP(
            global_dim, (self.context_dim,), self.context_dim
        ).to(self._device)
        self.si_decoder = ConditionalDecoder(
            self.embedding_dim + self.context_dim,
            self.decompress_dims,
            si_dim,
        ).to(self._device)
        self.si_context_encoder = MLP(
            si_dim, (self.context_dim,), self.context_dim
        ).to(self._device)
        self.o_decoder = ConditionalDecoder(
            self.embedding_dim + 2 * self.context_dim,
            self.decompress_dims,
            o_dim,
        ).to(self._device)

    def _all_modules(self):
        return [
            self.encoder,
            self.global_decoder,
            self.global_context_encoder,
            self.si_decoder,
            self.si_context_encoder,
            self.o_decoder,
        ]

    def save(self, filepath):
        state = {
            "encoder": self.encoder,
            "global_decoder": self.global_decoder,
            "global_context_encoder": self.global_context_encoder,
            "si_decoder": self.si_decoder,
            "si_context_encoder": self.si_context_encoder,
            "o_decoder": self.o_decoder,
            "global_transformer": self.global_transformer,
            "si_transformer": self.si_transformer,
            "o_transformer": self.o_transformer,
            "device": str(self._device),
            "config": {
                "embedding_dim": self.embedding_dim,
                "context_dim": self.context_dim,
            },
        }
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

        self._build_models(global_x.shape[1], si_x.shape[1], o_x.shape[1])
        global_tensor = torch.from_numpy(global_x).to(self._device)
        si_tensor = torch.from_numpy(si_x).to(self._device)
        o_tensor = torch.from_numpy(o_x).to(self._device)
        
        dataset = TensorDataset(
            global_tensor,
            si_tensor,
            o_tensor,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        parameters = []
        for module in self._all_modules():
            parameters.extend(module.parameters())
        optimizer = Adam(parameters, weight_decay=self.l2scale)
        scaler = torch.amp.GradScaler("cuda", enabled=self._device.type == "cuda")

        self.loss_values = []
        iterator = tqdm(range(self.epochs), disable=not self.verbose)
        for epoch in iterator:
            running = {"global": 0.0, "si": 0.0, "o": 0.0, "kl": 0.0}
            count = 0
            predicted_probability = self._context_probability(epoch)
            if self.kl_warmup_epochs > 0:
                current_kl_weight = self.kl_weight * min(
                    1.0, (epoch + 1) / float(self.kl_warmup_epochs)
                )
            else:
                current_kl_weight = self.kl_weight

            for global_b, si_b, o_b in loader:
                full_b = torch.cat([global_b, si_b, o_b], dim=1)
                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self._device.type == "cuda"):
                    mu, std, logvar = self.encoder(full_b)
                    z = mu + torch.randn_like(std) * std

                    global_raw, global_sigma = self.global_decoder(z)
                    global_pred = _activate_transformed(
                        global_raw, self.global_transformer.output_info_list
                    )
                    global_for_context = self._mix_context(
                        global_b, global_pred, predicted_probability
                    )
                    global_context = self.global_context_encoder(global_for_context)

                    si_raw, si_sigma = self.si_decoder(
                        torch.cat([z, global_context], dim=1)
                    )
                    si_pred = _activate_transformed(
                        si_raw, self.si_transformer.output_info_list
                    )
                    si_for_context = self._mix_context(
                        si_b, si_pred, predicted_probability
                    )
                    si_context = self.si_context_encoder(si_for_context)

                    o_raw, o_sigma = self.o_decoder(
                        torch.cat([z, global_context, si_context], dim=1)
                    )

                    global_loss = _block_reconstruction_loss(
                        global_raw,
                        global_b,
                        global_sigma,
                        self.global_transformer.output_info_list,
                        self.loss_factor,
                    )
                    si_loss = _block_reconstruction_loss(
                        si_raw,
                        si_b,
                        si_sigma,
                        self.si_transformer.output_info_list,
                        self.loss_factor,
                    )
                    o_loss = _block_reconstruction_loss(
                        o_raw,
                        o_b,
                        o_sigma,
                        self.o_transformer.output_info_list,
                        self.loss_factor,
                    )
                    kl_loss = -0.5 * torch.sum(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    ) / full_b.size(0)
                    loss = (
                        self.global_loss_weight * global_loss
                        + self.si_loss_weight * si_loss
                        + self.o_loss_weight * o_loss
                        + current_kl_weight * kl_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                scaler.step(optimizer)
                scaler.update()
                for decoder in (
                    self.global_decoder, self.si_decoder, self.o_decoder
                ):
                    decoder.sigma.data.clamp_(0.01, 1.0)

                bsz = full_b.size(0)
                count += bsz
                running["global"] += global_loss.detach().item() * bsz
                running["si"] += si_loss.detach().item() * bsz
                running["o"] += o_loss.detach().item() * bsz
                running["kl"] += kl_loss.detach().item() * bsz

            record = {
                "epoch": epoch + 1,
                "global_loss": running["global"] / count,
                "si_loss": running["si"] / count,
                "o_loss": running["o"] / count,
                "kl_loss": running["kl"] / count,
                "predicted_context_probability": predicted_probability,
                "kl_weight": current_kl_weight,
            }
            self.loss_values.append(record)
            total_display = (
                self.global_loss_weight * record["global_loss"]
                + self.si_loss_weight * record["si_loss"]
                + self.o_loss_weight * record["o_loss"]
                + current_kl_weight * record["kl_loss"]
            )
            iterator.set_description(
                f"Loss {total_display:.3f} | ctx {predicted_probability:.2f}"
            )

            if (epoch + 1) % 25 == 0:
                self.save(
                    os.path.join(
                        self.model_folder,
                        f"FactorizedVAE_checkpoint_epoch_{epoch + 1}.pkl",
                    )
                )

        pd.DataFrame(self.loss_values).to_csv(
            os.path.join(self.root_folder, "factorized_loss.csv"), index=False
        )

    @random_state
    def sample(self, samples, temperature=1.0, hard=True):
        for module in self._all_modules():
            module.eval()

        global_batches = []
        si_batches = []
        o_batches = []
        steps = int(np.ceil(samples / self.batch_size))
        with torch.no_grad():
            for _ in range(steps):
                z = torch.randn(
                    self.batch_size, self.embedding_dim, device=self._device
                )
                global_raw, global_sigma = self.global_decoder(z)
                global_x = _activate_transformed(
                    global_raw,
                    self.global_transformer.output_info_list,
                    temperature=temperature,
                    hard=hard,
                )
                global_context = self.global_context_encoder(global_x)

                si_raw, si_sigma = self.si_decoder(
                    torch.cat([z, global_context], dim=1)
                )
                si_x = _activate_transformed(
                    si_raw,
                    self.si_transformer.output_info_list,
                    temperature=temperature,
                    hard=hard,
                )
                si_context = self.si_context_encoder(si_x)

                o_raw, o_sigma = self.o_decoder(
                    torch.cat([z, global_context, si_context], dim=1)
                )
                o_x = _activate_transformed(
                    o_raw,
                    self.o_transformer.output_info_list,
                    temperature=temperature,
                    hard=hard,
                )

                global_batches.append(global_x.cpu().numpy())
                si_batches.append(si_x.cpu().numpy())
                o_batches.append(o_x.cpu().numpy())

        global_array = np.concatenate(global_batches, axis=0)[:samples]
        si_array = np.concatenate(si_batches, axis=0)[:samples]
        o_array = np.concatenate(o_batches, axis=0)[:samples]

        global_df = self.global_transformer.inverse_transform(
            global_array, self.global_decoder.sigma.detach().cpu().numpy()
        )
        si_df = self.si_transformer.inverse_transform(
            si_array, self.si_decoder.sigma.detach().cpu().numpy()
        )
        o_df = self.o_transformer.inverse_transform(
            o_array, self.o_decoder.sigma.detach().cpu().numpy()
        )
        return global_df, si_df, o_df

    def set_device(self, device):
        self._device = torch.device(device)
        for module in self._all_modules():
            module.to(self._device)

