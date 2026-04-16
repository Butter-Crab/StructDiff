import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from diffusion.SobelEdgeLoss import SobelEdgeLoss
# from diffusion.VGG_feature_extractor import FeatureExtractor

from common_utils.common import two_tuple
from diffusion.diffusion_utils import save_diffusion_sample, to_torch, linear_noise_schedule, cosine_noise_schedule


class Diffusion(LightningModule):
    def __init__(self, model, channels=3, timesteps=1000,
                 initial_lr=2e-4, training_target='x0', noise_schedule='cosine',
                 auto_sample=False, sample_every_n_steps=1000, sample_size=(32, 32), head_grid=None,
                 uses_positional_encoding=False):
        """Single-image DDPM wrapper used for StructDiff training and sampling."""
        super().__init__()
        self.save_hyperparameters(ignore=["model", "head_grid"])

        self.step_counter = 0
        self.auto_sample = auto_sample
        self.sample_every_n_steps = sample_every_n_steps
        self.sample_size = sample_size

        self.channels = channels
        self.model = model
        self.initial_lr = initial_lr
        self.training_target = training_target.lower()
        self.head_grid = head_grid
        self.uses_positional_encoding = uses_positional_encoding

        assert self.training_target in ['x0', 'noise']
        assert noise_schedule in ['linear', 'cosine']
        if noise_schedule == 'linear':
            betas = linear_noise_schedule(timesteps)
        else:
            betas = cosine_noise_schedule(timesteps)

        self.num_timesteps = int(betas.shape[0])
        alphas = 1. - betas
        alphas_hat = np.cumprod(alphas, axis=0)
        alphas_hat_prev = np.append(1., alphas_hat[:-1])

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_hat', to_torch(alphas_hat))
        self.register_buffer('alphas_hat_prev', to_torch(alphas_hat_prev))
        self.register_buffer('sqrt_alphas_hat', to_torch(np.sqrt(alphas_hat)))
        self.register_buffer('sqrt_one_minus_alphas_hat', to_torch(np.sqrt(1. - alphas_hat)))
        self.register_buffer('log_one_minus_alphas_hat', to_torch(np.log(1. - alphas_hat)))
        self.register_buffer('sqrt_recip_alphas_hat', to_torch(np.sqrt(1. / alphas_hat)))
        self.register_buffer('sqrt_recipm1_alphas_hat', to_torch(np.sqrt(1. / alphas_hat - 1)))
        posterior_variance = betas * (1. - alphas_hat_prev) / (1. - alphas_hat)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_hat_prev) / (1. - alphas_hat)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_hat_prev) * np.sqrt(alphas) / (1. - alphas_hat)))

    def _get_head_grid(self, batch_size, device):
        if self.head_grid is None:
            return None

        grid = self.head_grid.to(device)
        if grid.shape[0] == batch_size:
            return grid
        if grid.shape[0] == 1:
            return grid.expand(batch_size, -1, -1, -1)
        if grid.shape[0] > batch_size:
            return grid[:batch_size]

        repeat_count = (batch_size + grid.shape[0] - 1) // grid.shape[0]
        return grid.repeat(repeat_count, 1, 1, 1)[:batch_size]

    def predict_x0(self, x, t, clip_denoised=True, visualize_weights_and_features=False):
        batch_size = x.shape[0]
        if isinstance(t, torch.Tensor):
            t_tensor = t.to(x.device)
            sample_t = int(t_tensor[0].item())
        else:
            sample_t = int(t)
            t_tensor = torch.full((batch_size,), sample_t, dtype=torch.int64, device=x.device)

        grid = self._get_head_grid(batch_size, x.device)

        if self.training_target == 'x0':
            x_recon = self.model(
                x,
                grid,
                t_tensor,
                visualize_weights_and_features=visualize_weights_and_features,
                sample_t=sample_t,
                step_counter=self.step_counter,
            )
        else:
            noise_recon = self.model(x, grid, t_tensor)
            x_recon = self.predict_start_from_noise(x, t=sample_t, noise=noise_recon)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        return x_recon

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_hat[t] * x_t - self.sqrt_recipm1_alphas_hat[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised, visualize_weights_and_features=False):
        x_recon = self.predict_x0(
            x,
            t,
            clip_denoised=clip_denoised,
            visualize_weights_and_features=visualize_weights_and_features,
        )

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x[:,:3,:,:], t=t)
        
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_from_reconstruction(self, x, x_recon, t):
        model_mean, _, model_log_variance = self.q_posterior(
            x_start=x_recon,
            x_t=x[:, :self.channels, :, :],
            t=t,
        )
        noise = torch.randn_like(model_mean) if t > 0 else torch.zeros_like(model_mean)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample(self, x, t, visualize_weights_and_features=False, clip_denoised=True):
        x_recon = self.predict_x0(
            x,
            t,
            clip_denoised=clip_denoised,
            visualize_weights_and_features=visualize_weights_and_features,
        )
        return self.p_sample_from_reconstruction(x, x_recon, t)

    @torch.no_grad()
    def sample(self, image_size=(32, 32), batch_size=16, visualize_weights_and_features=False, custom_initial_img=None, custom_timesteps=None):
        """Run ancestral sampling from Gaussian noise or a custom latent."""
        image_size = two_tuple(image_size)
        sample_shape = (batch_size, self.channels, image_size[0], image_size[1])

        timesteps = custom_timesteps or self.num_timesteps
        img = custom_initial_img if custom_initial_img is not None else torch.randn(sample_shape, device=self.device)
        if img.shape[1] > self.channels:
            img = img[:, :self.channels, :, :]

        visualize_interval = max(1, timesteps // 3)
        for t in reversed(range(0, timesteps)):
            if t % visualize_interval == 0:
                img = self.p_sample(img, t, visualize_weights_and_features=visualize_weights_and_features)
            else:
                img = self.p_sample(img, t)

        return img


    @torch.no_grad()
    def sample_ddim(self, x_T=None, image_size=(32, 32), batch_size=16, sampling_step_size=100):
        """Run deterministic DDIM sampling."""
        seq = range(0, self.num_timesteps, sampling_step_size)
        seq_next = [-1] + list(seq[:-1])

        if x_T is None:
            image_size = two_tuple(image_size)
            sample_shape = (batch_size, self.channels, image_size[0], image_size[1])
            x_t = torch.randn(sample_shape, device=self.device)
        else:
            batch_size = x_T.shape[0] if len(x_T.shape) == 4 else 1
            x_t = x_T

        grid = self._get_head_grid(batch_size, x_t.device)
        zipped_reversed_seq = list(zip(reversed(seq), reversed(seq_next)))
        for t, t_next in zipped_reversed_seq:
            t_tensor = torch.full((batch_size,), t, dtype=torch.int64, device=self.device)
            e_t = self.model(x_t, grid, t_tensor)

            predicted_x0 = (x_t - self.sqrt_one_minus_alphas_hat[t] * e_t) / self.sqrt_alphas_hat[t]
            if t > 0:
                direction_to_x_t = self.sqrt_one_minus_alphas_hat[t_next] * e_t
                x_t = self.sqrt_alphas_hat[t_next] * predicted_x0 + direction_to_x_t
            else:
                x_t = predicted_x0

        return x_t

    def q_sample(self, x_start, t, noise=None):
        """Diffuse a clean image to timestep ``t``."""
        if noise is None:
            noise = torch.randn_like(x_start)

        return self.sqrt_alphas_hat[t] * x_start + self.sqrt_one_minus_alphas_hat[t] * noise
        

    def forward(self, x, *args, **kwargs):
        x = x.get('IMG')
        batch_size = x.shape[0]

        t = np.random.randint(0, self.num_timesteps)
        x_img = x[:,:3,:,:]
        mask = x[:,3:6,:,:]
        grid = x[:,6:,:,:] if x.shape[1] > 6 else None
        noise = torch.randn_like(x_img)
        x_noisy = self.q_sample(x_start=x_img, t=t, noise=noise)

        t_tensor = torch.full((batch_size, ), t, dtype=torch.int64, device=self.device)

        if self.training_target == 'x0':
            x0_recon = self.model(x_noisy, grid, t_tensor, training=True)
            mse_loss = F.mse_loss(x_img, x0_recon)
        else:
            noise_recon = self.model(x_noisy, grid, t_tensor)
            mse_loss = F.mse_loss(noise, noise_recon)

        mixed_image = torch.cat((x_img, mask), dim=1)
        edge_loss_fn = SobelEdgeLoss()
        edge_loss = edge_loss_fn(x0_recon, mixed_image)

        x0_recon_masked = x0_recon * mask
        x_img_masked = x_img * mask
        # The paper uses a foreground VGG term here. We keep the code path as a
        # commented reference and use foreground MSE instead because VGG is too slow.
        #
        # feature_extractor = FeatureExtractor().to(self.device)
        # criterion_content = torch.nn.L1Loss()
        # recon_features = feature_extractor(x0_recon_masked)
        # real_features = feature_extractor(x_img_masked)
        # foreground_recon_loss = criterion_content(recon_features, real_features.detach())

        foreground_recon_loss = F.mse_loss(x0_recon_masked, x_img_masked)

        mse_weight = 1.0
        sobel_weight = 0.2
        foreground_recon_weight = 0.2
        total_loss = (
            mse_weight * mse_loss
            + sobel_weight * edge_loss
            + foreground_recon_weight * foreground_recon_loss
        )

        return total_loss

    def training_step(self, batch, batch_idx):
        del batch_idx
        if self.auto_sample and self.step_counter % self.sample_every_n_steps == 0:
            sample = self.sample(image_size=self.sample_size, batch_size=1, visualize_weights_and_features=True)
            save_diffusion_sample(sample, f'{self.logger.log_dir}/sample_{self.step_counter}.png')

        loss = self.forward(batch)
        self.log('train_loss', loss)
        self.step_counter += 1
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[20], gamma=0.1, verbose=False)
        return [optim], [scheduler]
