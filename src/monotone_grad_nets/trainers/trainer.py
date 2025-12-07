import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, Callable, Dict, List
from ..utils.plotting import plot_train_metrics, plot_image_transport
try:
    import geomloss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False

try:
    from torch.func import vmap, jacrev, grad, functional_call
    TORCH_FUNC_AVAILABLE = True
except ImportError:
    TORCH_FUNC_AVAILABLE = False

class Trainer:
    """General-purpose trainer for M-MGN supporting multiple tasks."""
    def __init__(
        self,
        task: str = 'gradient',  # 'gradient' or 'optimal_transport'
        input_data : Optional[torch.Tensor] = None,
        target_data: Optional[torch.Tensor] = None,
        target_distribution: Optional[dict] = None,
        n_epochs: int = 50,
        lr: float = 0.01,
        criterion: str = 'L1loss',
        optimizer: str = 'Adam',
        weight_decay: float = 0,
        betas: Tuple[float, float] = (0.9, 0.999),
        model: Optional[nn.Module] = None,
        model_name: Optional[str] = None,
        true_fx: Optional[Callable] = None,
        batch_size: int = 32,
        device: str = 'cpu',
        verbose: bool = True,
        accumulation_steps: int = 1,
        use_amp: bool = False,
        grad_clip_max_norm: Optional[float] = None,
        plot: bool = False,
        sinkhorn_blur: float = 0.05,
        sinkhorn_scaling: float = 0.9,
        transport_cost_weight: float = 1.0,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None
    ):
        # Enhanced input validation
        if task not in ['gradient', 'optimal_transport']:
            raise ValueError(f"task must be 'gradient' or 'optimal_transport', got '{task}'")
        
        if model is None:
            raise ValueError("Model must be provided.")
        
        if criterion.lower() in ['kld', 'nll', 'sinkhorn'] and task != 'optimal_transport':
            raise ValueError(f"Criterion '{criterion}' requires task='optimal_transport'")
        
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")
        
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        
        if accumulation_steps < 1:
            raise ValueError(f"accumulation_steps must be >= 1, got {accumulation_steps}")
        
        if grad_clip_max_norm is not None and grad_clip_max_norm <= 0:
            raise ValueError(f"grad_clip_max_norm must be positive or None, got {grad_clip_max_norm}")
        
        if device not in ['cpu', 'cuda'] and not device.startswith('cuda:'):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:N', got '{device}'")
        
        self.task = task
        self.target_distribution = target_distribution
        self.device = device
        self.n_epochs = n_epochs
        self.verbose = verbose
        self.lr = lr
        self.batch_size = batch_size
        self.input_data = input_data
        self.target_data = target_data
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.grad_clip_max_norm = grad_clip_max_norm
        self.plot = plot
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_scaling = sinkhorn_scaling
        self.transport_cost_weight = transport_cost_weight
        self.scheduler_name = scheduler
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.metrics: Dict[str, List[float]] = {
            'train_loss': [],  'train_cost': []
        }

        # Initialize model
        self.model = model.to(device)
        self.model_name = model_name

        # Initialize the function to approximate (if applicable)
        self.true_fx = true_fx

        # Initialize optimizer and loss
        self.optimizer = self._get_optimizer(optimizer=optimizer, weight_decay=weight_decay, betas=betas)
        self.criterion_name = criterion
        self.criterion = self._get_criterion(criterion=criterion)
        self.scheduler = self._get_scheduler(self.scheduler_name, self.scheduler_kwargs)
        
        # Initialize GradScaler for AMP
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp and self.device != 'cpu' else None

    def _get_optimizer(self, optimizer: str, weight_decay: float, betas: Tuple[float, float]) -> optim.Optimizer:
        """Initialize the optimizer."""
        if optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.lr, betas=betas, weight_decay=weight_decay)
        elif optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported.")

    def _get_scheduler(self, scheduler: Optional[str], scheduler_kwargs: Dict) -> Optional[object]:
        """Initialize the learning rate scheduler."""
        if not scheduler:
            return None
        
        if scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, **scheduler_kwargs)
        elif scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **scheduler_kwargs)
        elif scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_kwargs)
        elif scheduler.lower() == 'exponential':
            return optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_kwargs)
        else:
            raise ValueError(f"Scheduler {scheduler} not supported.")

    def _get_criterion(self, criterion: str) -> Callable:
        """Initialize the criterion."""
        if criterion.lower() == 'l1loss':
            return nn.L1Loss()
        elif criterion.lower() in ['kld', 'nll', 'sinkhorn']:
            return lambda x, y: self._custom_loss(x, y)
        elif criterion.lower() == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Criterion '{criterion}' is not supported.")

    def _get_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gradient (model output)."""
        if self.model_name and self.model_name.lower() == 'icnn':
            # Compute gradients via autograd for ICNN
            outputs = self.model(x.squeeze(0))
            grad = torch.autograd.grad(
                outputs=outputs,
                inputs=x,
                grad_outputs=torch.ones_like(outputs),
                create_graph=True
            )[0]  # Extract gradient tensor
            return grad
        return self.model(x)

    def _compute_batch_jacobian(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute Jacobian for the entire batch efficiently.
        Uses torch.func.vmap and jacrev if available, otherwise falls back to loop.
        """
        if TORCH_FUNC_AVAILABLE:
            # Prepare functional call to handle stateful nn.Module
            params = dict(self.model.named_parameters())
            buffers = dict(self.model.named_buffers())
            
            if self.model_name and self.model_name.lower() == 'icnn':
                # ICNN outputs scalar potential, we want Hessian (Jacobian of gradient)
                def func_single(params, buffers, x):
                    # functional_call(module, (params, buffers), args)
                    out = functional_call(self.model, (params, buffers), (x.unsqueeze(0),))
                    return out.squeeze() # scalar

                # Hessian = jacrev(grad(f))
                # We need grad w.r.t x, so argnums=2
                # jacrev also w.r.t x
                
                # Define a function of x only, closing over params/buffers
                def f_x(x):
                    return func_single(params, buffers, x)
                
                # Compute Hessian per sample
                hessian_fn = jacrev(grad(f_x))
                batch_hessian_fn = vmap(hessian_fn)
                return batch_hessian_fn(x_batch)
            else:
                # MGN outputs gradient vector, we want Jacobian
                def func_single(params, buffers, x):
                    out = functional_call(self.model, (params, buffers), (x.unsqueeze(0),))
                    return out.squeeze(0) # vector

                def f_x(x):
                    return func_single(params, buffers, x)
                
                batch_jac_fn = vmap(jacrev(f_x))
                return batch_jac_fn(x_batch)
        else:
            # Fallback to loop
            batch_size = x_batch.shape[0]
            d = x_batch.shape[1]
            jacobian = torch.zeros(batch_size, d, d, device=x_batch.device)
            for i in range(batch_size):
                jacobian[i] = torch.autograd.functional.jacobian(
                    lambda x: self._get_grad(x.unsqueeze(0)), 
                    x_batch[i], 
                    create_graph=True
                )
            return jacobian

    def _kld_loss(self, g_x: torch.Tensor, mu_p: torch.Tensor, cov_p: torch.Tensor) -> torch.Tensor:
        """
        Compute the KLD between the empirical distribution of g_x and the target Gaussian.
        g_x is assumed to be a batch of samples with shape [batch_size, d].
        """
        # Ensure eps is the same dtype as g_x
        eps = torch.tensor(1e-6, dtype=g_x.dtype, device=self.device)
        
        # Estimate the empirical mean and covariance from g_x
        mu_q = g_x.mean(dim=0)
        diff = g_x - mu_q
        cov_q = (diff.T @ diff) / (g_x.size(0) - 1) + eps * torch.eye(g_x.shape[1], device=self.device, dtype=g_x.dtype)
        
        # Cast target parameters to match g_x dtype
        mu_p = mu_p.to(g_x.dtype)
        cov_p = cov_p.to(g_x.dtype) + eps * torch.eye(mu_p.shape[0], device=self.device, dtype=g_x.dtype)
        d = mu_q.shape[0]
        
        # Compute log determinants in a stable way
        _, logdet_q = torch.linalg.slogdet(cov_q)
        _, logdet_p = torch.linalg.slogdet(cov_p)
        
        epsilon = 1e-6  # Small positive value to prevent singularity
        inv_cov_p = torch.linalg.inv(cov_p + epsilon * torch.eye(cov_p.shape[0], device=cov_p.device))

        trace_term = torch.trace(inv_cov_p @ cov_q)
        mean_diff = (mu_p - mu_q).unsqueeze(0)  # shape [1, d]
        mean_term = mean_diff @ inv_cov_p @ mean_diff.T
        
        kld = 0.5 * (logdet_p - logdet_q - d + trace_term + mean_term.squeeze())
        return kld

    def _sinkhorn_loss(self, g_x: torch.Tensor, target_samples: torch.Tensor) -> torch.Tensor:
        """Compute Sinkhorn divergence using geomloss."""
        if not GEOMLOSS_AVAILABLE:
            raise ImportError("geomloss is required for Sinkhorn loss. Please install it.")
        
        loss_fn = geomloss.SamplesLoss(loss="sinkhorn", p=2, blur=self.sinkhorn_blur, scaling=self.sinkhorn_scaling, debias=True, backend="tensorized")
        return loss_fn(g_x, target_samples)

    def _custom_loss(self, x_batch: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        """Custom loss for optimal transport."""
        if self.criterion_name.lower() == 'sinkhorn':
            # Sinkhorn Loss + Transport Cost
            # We need target samples. x_batch is source. g_x is mapped source.
            # If target_distribution is provided as samples (tensor), we use it.
            # If it's a dict (Gaussian params), we sample from it.
            
            if self.target_data is not None:
                # Sample random batch from target data
                if self.target_data.shape[0] == 3:
                    target = self.target_data.permute(1, 2, 0).reshape(-1, 3)
                else:
                   # Change it to handle generic data (like flattened MNIST)
                    target = self.target_data

                idx = torch.randperm(target.size(0))[:g_x.size(0)]
                target_samples = target[idx].to(self.device)
            elif self.target_distribution:
                # Sample from Gaussian
                if 'weights' in self.target_distribution:
                     # Mixture of Gaussians sampling - simplified for now, assuming single or handled elsewhere
                     # For now raise error if not simple Gaussian or handled
                     raise NotImplementedError("Sinkhorn with MoG target distribution not yet fully implemented for sampling on the fly.")
                else:
                    mean = self.target_distribution['mean'].to(self.device)
                    cov = self.target_distribution['cov'].to(self.device)
                    target_samples = torch.distributions.MultivariateNormal(mean, cov).sample((g_x.size(0),))
            else:
                 # Standard Normal
                 target_samples = torch.randn_like(g_x)
            g_x = g_x.unsqueeze(0)              # [1, N, d]
            target_samples = target_samples.unsqueeze(0)  # [1, N, d]
            sinkhorn_term = self._sinkhorn_loss(g_x, target_samples)
            transport_term = self.compute_transport_cost(x_batch, g_x)
            
            return sinkhorn_term + self.transport_cost_weight * transport_term

        # Gaussian log-likelihood (NLL / KLD logic)
        if not self.target_distribution:
            # NLL for standard Gaussian: (d/2)log(2pi) + 0.5 * ||z||^2
            d = g_x.shape[1]
            const_term = (d / 2) * torch.log(torch.tensor(2 * torch.pi, device=g_x.device))
            log_p = -0.5 * (g_x ** 2).sum(dim=1) - const_term
            
            # Compute Jacobian for each sample in the batch
            # Use vectorized computation
            J = self._compute_batch_jacobian(x_batch)
                
            # Use slogdet for numerical stability
            # For monotone maps, determinant should be positive, so sign should be 1.
            sign, logabsdet = torch.linalg.slogdet(J)
            log_det = logabsdet
                
            return - (log_p + log_det).mean()
        elif 'weights' not in self.target_distribution:
            return self._kld_loss(g_x, self.target_distribution['mean'], self.target_distribution['cov'])

        else:
            m = self.target_distribution['mean']
            cov = self.target_distribution['cov']
            weights = self.target_distribution['weights']
            loss = float('inf')
            for i, w in enumerate(weights):
                kld = self._kld_loss(g_x, m[i], cov[i])
                if kld < loss:
                    loss = kld
            return loss

    def compute_transport_cost(self, x: torch.Tensor, g_x: torch.Tensor) -> torch.Tensor:
        """Compute Brenier's transport cost: E[||x - g(x)||^2]."""
        return torch.mean(torch.sum((x - g_x) ** 2, dim=1))

    def compute_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Jacobian of g(x) w.r.t. x."""
        return self._compute_batch_jacobian(x)

    def train(self, train_loader: DataLoader, plot_every: int = 20):
        """Main training loop."""
        pbar = tqdm(range(self.n_epochs), disable=not self.verbose)
        for epoch in pbar:
            self.model.train()
            train_loss = train_cost = 0
            self.optimizer.zero_grad() # Zero grad at start of epoch

            for i, batch in enumerate(train_loader): 
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    if self.task == 'gradient' and len(batch) > 1:
                        true_grad = batch[1]
                    else:
                        true_grad = None
                else:
                    x = batch
                    true_grad = None

                # Use device-aware autocast
                device_type = 'cuda' if self.device != 'cpu' else 'cpu'

                x = x.to(device_type, non_blocking=True)   

                # Forward pass
                x.requires_grad = True
                
                with torch.amp.autocast(device_type, enabled=self.use_amp):
                    g_x = self._get_grad(x)
                    
                    # Compute loss
                    if self.task == 'gradient':
                        loss = self.criterion(g_x, true_grad)
                    else:  # Optimal transport
                        loss = self.criterion(x, g_x)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulation_steps

                # Check for NaN/Inf loss
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at epoch {epoch}, batch {i}. Skipping batch.")
                    continue

                # Compute transport cost (outside AMP usually fine, but keep consistent if needed)
                with torch.no_grad():
                    cost = self.compute_transport_cost(x, g_x)

                # Backward pass and optimization
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                if (i + 1) % self.accumulation_steps == 0:
                    if self.scaler:
                        if self.grad_clip_max_norm:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        if self.grad_clip_max_norm:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()

                # Accumulate metrics (scale back up for reporting)
                train_loss += loss.item() * self.accumulation_steps * x.shape[0]
                train_cost += cost.item() * x.shape[0]

            # Flush remaining accumulated gradients at end of epoch
            if (i + 1) % self.accumulation_steps != 0:
                if self.scaler:
                    if self.grad_clip_max_norm:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.grad_clip_max_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if self.plot and self.input_data is not None and (epoch + 1) % plot_every == 0:
                self.model.eval()
                with torch.no_grad():
                    result = self.model(self.input_data.permute(1, 2, 0).view(-1, 3))
                result = result.view(*self.input_data.shape).cpu().clip(0, 1)
                
                plot_image_transport(
                        self.input_data.cpu(), 
                        result, 
                        self.target_data.cpu(),
                        title_suffix=f"(Epoch {epoch})"
                )

            # Step scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(train_loss / len(train_loader.dataset))
                else:
                    self.scheduler.step()

            # Update metrics
            self.metrics['train_loss'].append(train_loss / len(train_loader.dataset))
            self.metrics['train_cost'].append(train_cost / len(train_loader.dataset))

            # Update progress bar
            pbar.set_description(f"Epoch {epoch+1} | "
                                f"Train Loss: {train_loss / len(train_loader.dataset):.4f} | "
                                f"Train Cost: {train_cost / len(train_loader.dataset):.4f} | ")

    def plot_train_metrics(self, plot_cost: bool = False):
        """Plot training and validation metrics."""
        plot_train_metrics(self.metrics, plot_cost=plot_cost)