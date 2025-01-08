import torch
import onnx
import yaml
import os
from torch import nn
from torch.onnx import export
from argparse import ArgumentParser

from transformer import Embedding, AnalysisObjectTransformer
from losses import BCEDecorrelatedLoss

def load_training_config(model_dir: str) -> dict:
    with open(f"{model_dir}/training_config.yaml") as f:
        training_config = yaml.safe_load(f)
    return training_config

# Function to load the model from checkpoint and convert to ONNX
def convert_to_onnx(model_dir: str, output_path: str):
    
    config = load_training_config(model_dir)
    
    # Create dummy input data matching the shape of the expected inputs
    # For example, x could have shape (batch_size, seq_len, feature_size), etc.
    batch_size = 1000
    seq_len = config["sequence_length"]
    feature_size = 6
    
    X = torch.randn(batch_size, seq_len, feature_size)  # Example input tensor
    padding_mask = torch.ones(batch_size, seq_len)  # Example padding mask (1 for valid tokens, 0 for padding)
    attn_mask = torch.ones(batch_size, seq_len, seq_len)  # Example attention mask (usually square)
    dummy_input = (X, padding_mask)
    
    checkpoint = torch.load(os.path.join(model_dir, "best-checkpoint.ckpt"), weights_only=True)
    embedding = Embedding(
        input_dim = X.shape[-1], 
        embed_dims = config["embedding"]["embed_dims"], 
        normalize_input = config["embedding"]["normalize_input"],
    )
    model = AnalysisObjectTransformer(
        embedding = embedding,
        embed_dim = embedding.dim,
        output_dim = config["model"]["output_dim"],
        expansion_factor = config["model"]["expansion_factor"],
        encoder_layers = config["model"]["encoder_layers"],
        class_layers = config["model"]["class_layers"],
        dnn_layers = config["model"]["dnn_layers"],
        num_heads = config["model"]["num_heads"],
        hidden_activation = nn.GELU, 
        output_activation = None, #nn.Sigmoid,
        dropout = config["model"]["dropout"],
        loss_function = BCEDecorrelatedLoss(lam=config["loss_fn"]["lam"], weighted=config["loss_fn"]["weighted"]),
    )
    model.load_state_dict(checkpoint['state_dict'])  # Load only the model weights
    model.eval()  # Set the model to evaluation mode

    torch.onnx.export(model,  # model being run
                  (torch.unsqueeze(X[0], 0), torch.unsqueeze(padding_mask[0], 0)),  # Tuple of model inputs
                  output_path,  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=13,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input', 'mask'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes for each input
                                'mask': {0: 'batch_size'},  # variable length axes for each input
                                'output': {0: 'batch_size'}})

    print(f"Model successfully converted to ONNX and saved at {output_path}")


def main(args):
    convert_to_onnx(args.model_dir, args.output_path)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Convert a PyTorch model checkpoint to ONNX format.")
    parser.add_argument("model_dir", type=str, help="Directory of the model checkpoint file (e.g., best-checkpoint.ckpt)")
    parser.add_argument("output_path", type=str, help="Path to save the converted ONNX model (e.g., model.onnx)")
    args = parser.parse_args()
    
    main(args)

