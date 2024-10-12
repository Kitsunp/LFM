# Necessary imports for the model's functionality

import torch  # Core library for tensor operations and GPU computations
import torch.nn as nn  # PyTorch module containing classes for building neural networks
import torch.nn.functional as F  # Functional interface providing activation functions and other operations
from loguru import logger  # Advanced logging library for detailed runtime information
from torch.utils.checkpoint import checkpoint  # Functionality for Gradient Checkpointing to optimize memory usage
from torch.cuda.amp import autocast, GradScaler  # Tools for mixed precision training to accelerate computations and reduce memory
from typing import Tuple  # Utility for type annotations to improve code readability and maintenance

# Definition of the AdaptiveLinear class
class AdaptiveLinear(nn.Module):
    """
    Adaptive Linear Layer with dynamically adjustable weights and biases based on input.

    This layer combines static parameters with adaptive weights and biases generated from an additional input.
    It is beneficial in architectures requiring linear transformations that adapt dynamically to the input context.
    """

    def __init__(self, in_features: int, out_features: int, adapt_dim: int):
        """
        Initializes the AdaptiveLinear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            adapt_dim (int): Dimensionality of the adaptive input used to generate dynamic weights and biases.
        """
        super(AdaptiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Static parameters of the linear layer
        # 'weight' is a weight matrix of shape [out_features, in_features]
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        # 'bias' is a bias vector of shape [out_features]
        self.bias = nn.Parameter(torch.randn(out_features))

        # Layers to generate adaptive weights and biases based on 'adapt_dim'
        # 'adapt_weight' transforms the adaptive input into a vector that will be reshaped into a weight matrix
        self.adapt_weight = nn.Linear(adapt_dim, out_features * in_features)
        # 'adapt_bias' transforms the adaptive input into a bias vector
        self.adapt_bias = nn.Linear(adapt_dim, out_features)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AdaptiveLinear layer.

        Combines static weights with adaptive weights generated from 'adapt_input',
        applies the linear transformation to input 'x', and adds the adaptive bias.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size * seq_length, in_features].
            adapt_input (torch.Tensor): Adaptive input tensor with shape [batch_size * seq_length, adapt_dim].

        Returns:
            torch.Tensor: Output tensor after linear transformation with shape [batch_size * seq_length, out_features].
        """
        # Extract the batch size from the first dimension of x
        batch_size = x.size(0)

        # Generate adaptive weights from 'adapt_input'
        adapt_weight = self.adapt_weight(adapt_input)  # Shape: [batch_size, out_features * in_features]
        # Reshape adaptive weights to [batch_size, out_features, in_features]
        adapt_weight = adapt_weight.view(batch_size, self.out_features, self.in_features)

        # Generate adaptive biases from 'adapt_input'
        adapt_bias = self.adapt_bias(adapt_input)  # Shape: [batch_size, out_features]

        # Combine static weights with adaptive weights
        # Expand static weight to [1, out_features, in_features] and add adaptive weights
        weight = self.weight.unsqueeze(0) + adapt_weight  # Shape: [batch_size, out_features, in_features]

        # Prepare x for batch matrix multiplication by adding a dimension
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, in_features]

        # Perform batch matrix multiplication between x and the transposed combined weights
        output = torch.bmm(x, weight.transpose(1, 2))  # Shape: [batch_size, 1, out_features]
        # Remove the added dimension to get [batch_size, out_features]
        output = output.squeeze(1)  # Shape: [batch_size, out_features]

        # Add the adaptive bias to the output
        output += adapt_bias  # Shape: [batch_size, out_features]

        return output

# Definition of the TokenMixing class
class TokenMixing(nn.Module):
    """
    Token Mixing Layer that performs token-wise interactions using AdaptiveLinear layers.

    Operates across the sequence dimension (sequence_length), allowing tokens to interact dynamically
    based on provided adaptive information.
    """

    def __init__(self, token_dim: int, adapt_dim: int):
        """
        Initializes the TokenMixing layer.

        Args:
            token_dim (int): Dimensionality of the tokens (embedding_dim).
            adapt_dim (int): Dimensionality of the adaptive input used to generate dynamic weights and biases.
        """
        super(TokenMixing, self).__init__()
        # AdaptiveLinear layer to mix tokens
        self.token_mixing = AdaptiveLinear(token_dim, token_dim, adapt_dim)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the TokenMixing layer.

        Transforms each token in the sequence using an adaptive linear layer that considers adaptive information.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length, embedding_dim].
            adapt_input (torch.Tensor): Adaptive input tensor with shape [batch_size, adapt_dim].

        Returns:
            torch.Tensor: Output tensor after token mixing with shape [batch_size, sequence_length, embedding_dim].
        """
        # Extract dimensions from input tensor
        batch_size, seq_length, embed_dim = x.shape
        # Reshape x to [batch_size * sequence_length, embedding_dim] for processing
        x = x.view(batch_size * seq_length, embed_dim)  # Shape: [batch_size * seq_length, embedding_dim]

        # Efficiently expand 'adapt_input' for each token in the sequence without using repeat
        # Add a dimension for sequence and expand
        adapt_input = adapt_input.unsqueeze(1).expand(-1, seq_length, -1).contiguous()
        # Reshape to [batch_size * sequence_length, adapt_dim]
        adapt_input = adapt_input.view(batch_size * seq_length, -1)  # Shape: [batch_size * seq_length, adapt_dim]

        # Apply the adaptive token mixing layer
        x_mixed = self.token_mixing(x, adapt_input)  # Shape: [batch_size * seq_length, token_dim]
        # Reshape back to [batch_size, sequence_length, embedding_dim]
        return x_mixed.view(batch_size, seq_length, embed_dim)  # Shape: [batch_size, sequence_length, embedding_dim]

# Definition of the ChannelMixing class
class ChannelMixing(nn.Module):
    """
    Channel Mixing Layer that performs cross-channel interactions using AdaptiveLinear layers.

    Operates across the embedding dimension (embedding_dim), allowing channels to interact dynamically
    based on provided adaptive information.
    """

    def __init__(self, channel_dim: int, adapt_dim: int):
        """
        Initializes the ChannelMixing layer.

        Args:
            channel_dim (int): Dimensionality of the channels (embedding_dim).
            adapt_dim (int): Dimensionality of the adaptive input used to generate dynamic weights and biases.
        """
        super(ChannelMixing, self).__init__()
        # AdaptiveLinear layer to mix channels
        self.channel_mixing = AdaptiveLinear(channel_dim, channel_dim, adapt_dim)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ChannelMixing layer.

        Transforms each channel in the embedding dimension using an adaptive linear layer that considers adaptive information.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length, embedding_dim].
            adapt_input (torch.Tensor): Adaptive input tensor with shape [batch_size, adapt_dim].

        Returns:
            torch.Tensor: Output tensor after channel mixing with shape [batch_size, sequence_length, embedding_dim].
        """
        # Extract dimensions from input tensor
        batch_size, seq_length, embed_dim = x.shape
        # Reshape x to [batch_size * sequence_length, embedding_dim] for processing
        x = x.view(batch_size * seq_length, embed_dim)  # Shape: [batch_size * seq_length, embedding_dim]

        # Efficiently expand 'adapt_input' for each channel in the sequence without using repeat
        # Add a dimension for sequence and expand
        adapt_input = adapt_input.unsqueeze(1).expand(-1, seq_length, -1).contiguous()
        # Reshape to [batch_size * sequence_length, adapt_dim]
        adapt_input = adapt_input.view(batch_size * seq_length, -1)  # Shape: [batch_size * seq_length, adapt_dim]

        # Apply the adaptive channel mixing layer
        x_mixed = self.channel_mixing(x, adapt_input)  # Shape: [batch_size * seq_length, channel_dim]
        # Reshape back to [batch_size, sequence_length, embedding_dim]
        return x_mixed.view(batch_size, seq_length, embed_dim)  # Shape: [batch_size, sequence_length, embedding_dim]

# Definition of the MixtureOfExperts class
class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) Module that dynamically selects experts based on input.

    Implements Sparse Activation and Expert Balancing to ensure equitable usage of available experts.
    This module operates after channel and token mixing layers.
    """

    def __init__(self, expert_dim: int, num_experts: int, adapt_dim: int, top_k: int = 2, balance_loss_weight: float = 1e-3):
        """
        Initializes the MixtureOfExperts module.

        Args:
            expert_dim (int): Dimensionality of the experts.
            num_experts (int): Total number of available experts.
            adapt_dim (int): Dimensionality of the adaptive input used to generate dynamic weights and biases.
            top_k (int, optional): Number of experts to select per input. Defaults to 2.
            balance_loss_weight (float, optional): Weight of the balance loss for expert balancing. Defaults to 1e-3.
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight

        # Create a list of adaptive experts
        # Each expert is an instance of AdaptiveLinear
        self.experts = nn.ModuleList([
            AdaptiveLinear(expert_dim, expert_dim, adapt_dim)
            for _ in range(num_experts)
        ])
        # Define the gating layer to determine expert selection
        self.gating = nn.Linear(adapt_dim, num_experts)

    def forward(self, x: torch.Tensor, adapt_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MixtureOfExperts module.

        Dynamically selects experts based on adaptive input, combines their outputs, and computes a balance loss
        to ensure equitable expert usage.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length, embedding_dim].
            adapt_input (torch.Tensor): Adaptive input tensor with shape [batch_size, adapt_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Combined output tensor with shape [batch_size, sequence_length, embedding_dim].
                - Balance loss scalar to encourage equitable expert usage.
        """
        # Extract dimensions from input tensor
        batch_size, seq_length, embed_dim = x.shape
        # Reshape x to [batch_size * sequence_length, embedding_dim] for processing
        x = x.view(batch_size * seq_length, embed_dim)  # Shape: [batch_size * seq_length, embedding_dim]

        # Efficiently expand 'adapt_input' for each token in the sequence without using repeat
        # Add a dimension for sequence and expand
        adapt_input = adapt_input.unsqueeze(1).expand(-1, seq_length, -1).contiguous()
        # Reshape to [batch_size * sequence_length, adapt_dim]
        adapt_input = adapt_input.view(batch_size * seq_length, -1)  # Shape: [batch_size * seq_length, adapt_dim]

        # Generate gating scores from 'adapt_input' to determine expert selection
        gate_scores = self.gating(adapt_input)  # Shape: [batch_size * seq_length, num_experts]
        # Apply softmax to obtain gating probabilities
        gate_probs = F.softmax(gate_scores, dim=-1)  # Shape: [batch_size * seq_length, num_experts]

        # Select the top_k experts based on gating probabilities
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)  # Both shapes: [batch_size * seq_length, top_k]

        # Create a mask to indicate selected experts
        mask = torch.zeros_like(gate_probs)  # Shape: [batch_size * seq_length, num_experts]
        # Scatter 1.0 into the mask at the positions of the top_k selected experts
        mask.scatter_(1, top_k_indices, 1.0)  # Shape: [batch_size * seq_length, num_experts]

        # Calculate balance loss to ensure equitable usage of experts
        # Expected usage per expert = (batch_size * seq_length * top_k) / num_experts
        expected_usage = (batch_size * seq_length * self.top_k) / self.num_experts
        # Actual usage per expert by summing the mask across the batch and sequence
        usage = mask.sum(dim=0)  # Shape: [num_experts]
        # Compute the squared difference between actual and expected usage
        balance_loss = (usage - expected_usage) ** 2
        # Sum all differences to get the total balance loss
        balance_loss = balance_loss.sum()

        # Initialize the output tensor as zeros with the same shape as x
        output = torch.zeros_like(x)  # Shape: [batch_size * seq_length, expert_dim]

        # Iterate over each expert to process the selected inputs
        for i, expert in enumerate(self.experts):
            # Select inputs that have chosen expert i
            selected = mask[:, i].unsqueeze(1)  # Shape: [batch_size * seq_length, 1]
            if selected.sum() == 0:
                continue  # Skip if no inputs have selected this expert

            # Multiply x by the mask to zero out inputs not selected by this expert
            selected_x = x * selected  # Shape: [batch_size * seq_length, expert_dim]
            # Pass the selected inputs through the adaptive expert
            expert_output = expert(selected_x, adapt_input)  # Shape: [batch_size * seq_length, expert_dim]
            # Accumulate the expert's output into the final output tensor
            output += expert_output  # Shape: [batch_size * seq_length, expert_dim]

        # Reshape the output back to [batch_size, sequence_length, embedding_dim]
        output = output.view(batch_size, seq_length, embed_dim)  # Shape: [batch_size, sequence_length, embedding_dim]

        return output, balance_loss

# Definition of the LFModel class
class LFModel(nn.Module):
    """
    Custom LF Model architecture that combines Token Mixing, Channel Mixing, and Mixture of Experts (MoE).

    Accepts a 3D input tensor with shape [batch_size, sequence_length, embedding_dim].
    Implements Gradient Checkpointing for memory optimization during training.
    """

    def __init__(self, token_dim: int, channel_dim: int, expert_dim: int, adapt_dim: int, num_experts: int, top_k: int = 2, balance_loss_weight: float = 1e-3):
        """
        Initializes the LFModel.

        Args:
            token_dim (int): Dimensionality of the tokens.
            channel_dim (int): Dimensionality of the channels.
            expert_dim (int): Dimensionality of the experts in the MoE.
            adapt_dim (int): Dimensionality of the adaptive input.
            num_experts (int): Number of experts in the MoE.
            top_k (int, optional): Number of experts to select per input in the MoE. Defaults to 2.
            balance_loss_weight (float, optional): Weight of the balance loss in the MoE. Defaults to 1e-3.
        """
        super(LFModel, self).__init__()
        # Featurizer layer to transform input tokens into adaptive features
        self.featurizer = nn.Linear(token_dim, adapt_dim)
        # Token mixing layer
        self.token_mixer = TokenMixing(token_dim, adapt_dim)
        # Channel mixing layer
        self.channel_mixer = ChannelMixing(channel_dim, adapt_dim)
        # Mixture of Experts module
        self.moe = MixtureOfExperts(expert_dim, num_experts, adapt_dim, top_k, balance_loss_weight)
        # Output layer to transform MoE output back to token dimension
        self.output_layer = nn.Linear(expert_dim, token_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the LFModel.

        Processes the input through featurization, token mixing, channel mixing, and Mixture of Experts stages,
        and generates the final output. Utilizes Gradient Checkpointing to optimize memory usage during training.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, sequence_length, embedding_dim].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Final output tensor with shape [batch_size, sequence_length, token_dim].
                - Balance loss scalar from the MoE module.
        """
        # Log the shape of the input for debugging purposes
        logger.info("Input shape: {}", x.shape)

        # Featurization stage
        # Compute the mean of x across the sequence dimension to obtain a summary for adaptive input
        adapt_input = self.featurizer(x.mean(dim=1))  # Shape: [batch_size, adapt_dim]
        logger.info("Featurization complete. Shape: {}", adapt_input.shape)

        # Token Mixing with Gradient Checkpointing
        # Apply token mixing layer while optimizing memory usage
        token_mixed = checkpoint(self.token_mixer, x, adapt_input)  # Shape: [batch_size, sequence_length, embedding_dim]
        logger.info("Token mixing complete. Shape: {}", token_mixed.shape)

        # Channel Mixing with Gradient Checkpointing
        # Apply channel mixing layer while optimizing memory usage
        channel_mixed = checkpoint(self.channel_mixer, token_mixed, adapt_input)  # Shape: [batch_size, sequence_length, embedding_dim]
        logger.info("Channel mixing complete. Shape: {}", channel_mixed.shape)

        # Mixture of Experts with Gradient Checkpointing
        # Apply MoE module while optimizing memory usage and retrieve balance loss
        expert_output, balance_loss = checkpoint(self.moe, channel_mixed, adapt_input)  # Shape: [batch_size, sequence_length, embedding_dim], scalar
        logger.info("Mixture of Experts complete. Shape: {}", expert_output.shape)

        # Final Output
        # Transform the MoE output back to the token dimension
        output = self.output_layer(expert_output)  # Shape: [batch_size, sequence_length, token_dim]
        logger.info("Output shape: {}", output.shape)

        return output, balance_loss

# Summary of Class Differences and Definitions

"""
Summary of Class Differences and Definitions

1. `AdaptiveLinear`:
   - **Role**: Serves as the foundational adaptive layer that dynamically adjusts its weights and biases based on an additional input.
   - **Usage**: Used by `TokenMixing`, `ChannelMixing`, and each expert within `MixtureOfExperts` to enable context-aware transformations.

2. `TokenMixing`:
   - **Role**: Enables dynamic interactions between tokens within a sequence, capturing dependencies and contextual relationships.
   - **Usage**: Applies adaptive token transformations to enhance sequence modeling capabilities.

3. `ChannelMixing`:
   - **Role**: Facilitates dynamic interactions between different channels or features within the embedding dimension, capturing inter-feature dependencies.
   - **Usage**: Applies adaptive channel transformations to enrich feature representations.

4. `MixtureOfExperts`:
   - **Role**: Implements a dynamic expert selection mechanism, allowing the model to leverage specialized experts based on input context.
   - **Usage**: Enhances model capacity and flexibility by selecting and combining outputs from multiple experts, while ensuring balanced expert utilization.

5. `LFModel`:
   - **Role**: Integrates all the aforementioned components into a cohesive architecture, orchestrating the flow of data through featurization, token mixing, channel mixing, and Mixture of Experts.
   - **Usage**: Acts as the main model class that processes input data and generates the final output, optimized for memory efficiency through Gradient Checkpointing.

Additional Notes:

- **Gradient Checkpointing**:
  - **What It Is**: A memory optimization technique that reduces memory usage by selectively storing intermediate activations and recomputing others during backpropagation.
  - **Benefits**: Enables training larger models on GPUs with limited memory by lowering memory consumption at the cost of increased computation time.
  - **Implementation**: Utilized in the `forward` method of `LFModel` for the `TokenMixing`, `ChannelMixing`, and `MixtureOfExperts` layers.

- **Logging with Loguru**:
  - **What It Is**: An advanced logging library that simplifies logging operations and provides a user-friendly interface.
  - **Usage in the Code**: Incorporated within the `LFModel` class to log the shapes of tensors at various stages of processing. This aids in debugging and ensures tensor dimensions align as expected.

- **Dynamic Adaptability**:
  - **Concept**: The ability of the model to adjust its parameters (weights and biases) dynamically based on contextual or adaptive inputs.
  - **Advantages**:
    - **Context-Aware Transformations**: Allows the model to tailor its transformations to specific input contexts, enhancing performance on diverse tasks.
    - **Improved Generalization**: Facilitates better handling of varying data patterns by dynamically adjusting internal parameters.
  - **Implementation in the Code**: Achieved through the `AdaptiveLinear` layers, which generate dynamic weights and biases based on the `adapt_input`.

- **Expert Balancing in MoE**:
  - **Importance**: Prevents some experts from being overused while others remain underutilized, ensuring efficient resource utilization and balanced learning.
  - **Balance Loss**: Calculated in the `MixtureOfExperts` class to penalize deviations from expected expert usage, promoting equitable distribution of workload across experts.
  - **Adjustability**: The weight of the balance loss (`balance_loss_weight`) can be tuned to prioritize balance versus model performance.

- **Model Flexibility and Scalability**:
  - **Mixture of Experts (MoE)**: Allows the model to scale its capacity by adding more experts without a proportional increase in computational cost per input.
  - **Adaptive Layers**: Provide the model with the flexibility to adjust transformations dynamically, making it suitable for tasks requiring context-sensitive processing.

This integrated documentation within the code provides a thorough understanding of each component, their interactions, and the overall architecture of the `LFModel`. By leveraging adaptive transformations and a Mixture of Experts framework, the model is designed to be both flexible and scalable, capable of handling complex and diverse data patterns efficiently.
"""

