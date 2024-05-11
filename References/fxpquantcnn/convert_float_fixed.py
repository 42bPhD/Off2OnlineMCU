
import torch

class ConvertFloatFixed:
    """Class to convert floating point numbers to fixed point representations
        
    Args:
        bitwidth (integer): Bitwidth of the Fixed-point representation
        fractional_bits (integer): Fractional offset of fixed-point representation
        base (int, optional): Base for scale. Defaults to 2.
    """

    _sign_bit = 1

    def __init__(self, bitwidth, fractional_bits, base=2):
        self.base = base
        self.bitwidth = bitwidth
        self.fractional_bits = fractional_bits

    def __call__(self, input_tensor):
        return self.quantize(input_tensor)

    @property
    def integer_bits(self):
        """Number of integer bits in fxp-representation
        """
        return self.bitwidth - self.fractional_bits - self._sign_bit

    @property
    def scale(self):
        """Scale to shift the numbers by based on fractional offset F
        """
        return self.base ** float(self.fractional_bits)
    
    @property
    def max_value(self):
        """Maximum value based on bitwidth - 1
        """
        if self.bitwidth == self._sign_bit:
            return 0
        return (self.base ** (self.bitwidth - self._sign_bit)) - 1

    @property
    def min_value(self):
        """Minimum value based on bitwidth - 1
        """
        if self.bitwidth == self._sign_bit:
            return 0
        return - (self.base ** (self.bitwidth - self._sign_bit) - 1)

    def quantize(self, input_tensor):
        """Quantize the given set of numbers to representation specified
        
        Args:
            input_tensor (Tensor): input tensor to quantize
        Returns:
            Tensor: output tensor of quantized numbers for the specified representation
        """
        
        rounded_tensor = torch.round(input_tensor * self.scale)
        clipped_tensor = torch.clamp(rounded_tensor, self.min_value, self.max_value)
        output_tensor = clipped_tensor / self.scale

        return output_tensor
    
    def quantize_sqrt(self, input_tensor):
        """Quantize the given set of numbers to representation specified 
        non-uniformly using square-root
        
        Args:
            input_tensor (Tensor): input tensor to quantize
        Returns:
            Tensor: output tensor of quantized numbers for the specified representation
        """
        sqrt_tensor = torch.sign(input_tensor) * torch.sqrt(torch.abs(input_tensor))
        rounded_tensor = torch.round(sqrt_tensor * self.scale)
        clipped_tensor = torch.clamp(rounded_tensor, self.min_value, self.max_value)
        unscaled_tensor = clipped_tensor / self.scale
        output_tensor = torch.sign(unscaled_tensor) * (unscaled_tensor ** 2)

        return output_tensor

    def quantize_pt(self, x):
        """PyTorch version of function for quantization to fixed-point
        
        Args:
            x (Tensor): Input tensor of values to quantize
        Returns:
            Tensor: Output tensor of quantized values
        """
        int_val = x * self.scale
        integer_value = torch.round(int_val)
        clip_value = torch.clamp(integer_value, self.min_value, self.max_value)
        y = clip_value / self.scale

        return y

