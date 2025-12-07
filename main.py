import argparse
import os
import torch
import glob
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from PIL import Image
from typing import Optional, Tuple, Dict

from monotone_grad_nets.trainers.trainer import Trainer
from monotone_grad_nets.models.icnn import I_CNN
from monotone_grad_nets.models.mmgn import M_MGN
from monotone_grad_nets.models.cmgn import C_MGN

# --- Data Loading Utilities ---

def load_image_as_pixels(path: str, transform: transforms.Compose) -> torch.Tensor:
    """Loads an image or folder of images and flattens them to [N, C] pixels."""
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = glob.glob(os.path.join(path, "*.*"))
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        raise ValueError(f"No images found at {path}")

    pixel_tensors = []
    print(f"Loading {len(files)} images from {path}...")
    
    for f in files:
        img = Image.open(f).convert("RGB")
        img_t = transform(img) # [3, H, W]
        # Flatten to [H*W, 3]
        pixels = img_t.permute(1, 2, 0).reshape(-1, 3)
        pixel_tensors.append(pixels)
    
    # Concatenate all pixels from all images
    return torch.cat(pixel_tensors, dim=0)

def get_data_loader(
    path: str, 
    task: str, 
    batch_size: int, 
    transform: Optional[transforms.Compose] = None
) -> Tuple[DataLoader, int, torch.Tensor]:
    """
    Returns:
        loader: DataLoader for training
        input_dim: Dimension of the data
        viz_batch: A static batch for visualization purposes
    """
    if task == 'optimal_transport':
        # Expecting images or tensors of points
        if path.endswith('.pt'):
            data = torch.load(path)
            # Assume data is tensor [N, D]
            if isinstance(data, tuple): data = data[0] # Handle (x, y) case by taking x
        else:
            # Assume Images
            if transform is None:
                raise ValueError("Transform required for image loading")
            data = load_image_as_pixels(path, transform)
            
        input_dim = data.shape[1]
        dataset = TensorDataset(data)
        viz_batch = data[:min(len(data), 1000)] # Take first 1000 for viz
        
    elif task == 'gradient':
        # Expecting .pt file with (x, grad_x)
        if not path.endswith('.pt'):
             raise ValueError("Gradient task requires .pt file with (x, true_grad) tuple.")
        
        data = torch.load(path)
        if isinstance(data, (list, tuple)) and len(data) == 2:
            x, grads = data
        else:
            raise ValueError(f"Gradient dataset must contain (x, grads) tuple. Got {type(data)}")

        input_dim = x.shape[1]
        dataset = TensorDataset(x, grads)
        viz_batch = x[:min(len(x), 1000)]
        
    else:
        raise ValueError(f"Unknown task: {task}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader, input_dim, viz_batch

def get_target_stats(path: Optional[str], transform: transforms.Compose) -> Optional[Dict]:
    """Computes Mean and Covariance of target image(s) for Color Transport."""
    if not path:
        return None
    
    print(f"Computing target statistics from {path}...")
    target_pixels = load_image_as_pixels(path, transform)
    
    # Compute stats
    mean = target_pixels.mean(dim=0)
    cov = torch.cov(target_pixels.T)
    
    return {'mean': mean, 'cov': cov}

# --- Model Factory ---

def get_model(name: str, input_dim: int, hidden_dim: int, num_layers: int):
    if name == 'icnn':
        return I_CNN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif name == 'mmgn':
        return M_MGN(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers)
    elif name == 'cmgn':
        return C_MGN(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=input_dim, 
            num_layers=num_layers
        )
    else:
        raise ValueError(f"Unknown model: {name}")

# --- Commands ---

def train(args):
    print(f"--- Starting Training ({args.task}) ---")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Transforms (only used if loading images)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # 1. Load Data
    train_loader, input_dim, viz_batch = get_data_loader(
        args.dataset_path, 
        args.task, 
        args.batch_size, 
        transform
    )
    print(f"Data Loaded: Input Dim={input_dim}, Batches={len(train_loader)}")

    # 2. Prepare Target Distribution (for Optimal Transport)
    target_dist = None
    target_viz = None
    if args.task == 'optimal_transport':
        if args.target_path:
            target_dist = get_target_stats(args.target_path, transform)
            # Load a snippet of target data for visualization plotting
            target_viz = load_image_as_pixels(args.target_path, transform)[:1000]
        else:
            print("No target path provided. Using Standard Normal distribution.")

    # 3. Initialize Model
    model = get_model(args.model, input_dim, args.hidden_dim, args.num_layers).to(device)
    
    # 4. Initialize Trainer
    trainer = Trainer(
        task=args.task,
        model=model,
        model_name=args.model,
        input_data=viz_batch.to(device),
        target_data=target_viz.to(device) if target_viz is not None else None,
        target_distribution=target_dist, # Pass the dict (mean, cov)
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=str(device),
        accumulation_steps=args.accum_steps,
        plot=not args.no_plot, # Plotting enabled by default
        verbose=True
    )

    # 5. Run
    trainer.train(train_loader)
    
    # 6. Save
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.model}_{args.task}.pt")
    
    # Save extra metadata for inference (input_dim is crucial)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(args),
        'input_dim': input_dim
    }, save_path)
    
    print(f"Model and config saved to {save_path}")

def test(args):
    print(f"--- Starting Inference ---")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. Load Checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    input_dim = checkpoint.get('input_dim', 3) # Default to 3 if missing
    
    print(f"Loading {config['model']} (Input Dim: {input_dim})...")
    
    model = get_model(
        config['model'], 
        input_dim=input_dim, 
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Load Input
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor(),
    ])
    
    image = Image.open(args.input_path).convert("RGB")
    x_tensor = transform(image).to(device) # [3, H, W]
    c, h, w = x_tensor.shape
    
    # Flatten: [H*W, C]
    x_flat = x_tensor.permute(1, 2, 0).reshape(-1, c)
    
    # 3. Process
    print("Processing...")
    with torch.no_grad():
        # Process in chunks to avoid OOM on large images
        chunk_size = 4096 * 4
        outputs = []
        for i in range(0, x_flat.shape[0], chunk_size):
            batch = x_flat[i : i + chunk_size]
            out_batch = model(batch)
            outputs.append(out_batch)
        
        out_flat = torch.cat(outputs, dim=0)
    
    # 4. Reshape and Save
    out_tensor = out_flat.view(h, w, c).permute(2, 0, 1).cpu()
    out_img = transforms.ToPILImage()(out_tensor.clamp(0, 1))
    
    os.makedirs(args.output_path, exist_ok=True)
    fname = os.path.basename(args.input_path)
    save_path = os.path.join(args.output_path, f"transformed_{fname}")
    
    out_img.save(save_path)
    print(f"Saved transformed image to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Monotone Gradient Networks CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Train Parser ---
    train_parser = subparsers.add_parser("train", help="Train a model")
    
    # Core
    train_parser.add_argument("--task", type=str, default="optimal_transport", choices=["optimal_transport", "gradient"])
    train_parser.add_argument("--model", type=str, default="mmgn", choices=["icnn", "mmgn", "cmgn"])
    train_parser.add_argument("--device", type=str, default="cuda:0")
    
    # Data
    train_parser.add_argument("--dataset_path", type=str, required=True, help="Path to images (OT) or .pt file (Gradient)")
    train_parser.add_argument("--target_path", type=str, help="Path to target image(s) for OT reference")
    train_parser.add_argument("--output_dir", type=str, default="checkpoints")
    
    # Hyperparams
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=1024)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--hidden_dim", type=int, default=64)
    train_parser.add_argument("--num_layers", type=int, default=5)
    train_parser.add_argument("--accum_steps", type=int, default=1)
    
    # Misc
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--no_plot", action="store_true", help="Disable plotting during training")

    # --- Test Parser ---
    test_parser = subparsers.add_parser("test", help="Inference / Image Translation")
    test_parser.add_argument("--checkpoint", type=str, required=True)
    test_parser.add_argument("--input_path", type=str, required=True, help="Image to transform")
    test_parser.add_argument("--output_path", type=str, default="results")
    test_parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    # Set seed
    if 'seed' in args:
        torch.manual_seed(args.seed)

    if args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)

if __name__ == "__main__":
    main()