import torch
import torch.nn as nn

# --- Configuration Defaults (for test function) ---
# These should ideally match the input expected by your pipeline
IMG_HEIGHT = 60
IMG_WIDTH = 160

class DdddOcr(nn.Module):
    """
    CNN Feature Extractor Backbone.
    Based on the CRNN discussion for captcha recognition, but only provides the CNN part.
    Outputs a feature map intended to be fed into a subsequent sequence model (like RNN).
    Designed to be a potential replacement for the original DdddOcr class definition.
    """
    def __init__(self, nc=1, leakyRelu=False): # Default nc=1 for grayscale
        super(DdddOcr, self).__init__()

        self.cnn = nn.Sequential()

        # Layer 1: Input (B, nc, 60, 160) -> Output (B, 64, 30, 80)
        self.cnn.add_module('conv0', nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1))
        if leakyRelu:
            self.cnn.add_module('relu0', nn.LeakyReLU(0.2, inplace=True))
        else:
            self.cnn.add_module('relu0', nn.ReLU(True))
        self.cnn.add_module('pool0', nn.MaxPool2d(kernel_size=2, stride=2))

        # Layer 2: Input (B, 64, 30, 80) -> Output (B, 128, 15, 40)
        self.cnn.add_module('conv1', nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1))
        if leakyRelu:
            self.cnn.add_module('relu1', nn.LeakyReLU(0.2, inplace=True))
        else:
            self.cnn.add_module('relu1', nn.ReLU(True))
        self.cnn.add_module('pool1', nn.MaxPool2d(kernel_size=2, stride=2))

        # Layer 3: Input (B, 128, 15, 40) -> Output (B, 256, 7, 41)
        self.cnn.add_module('conv2', nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('batchnorm2', nn.BatchNorm2d(256))
        if leakyRelu:
            self.cnn.add_module('relu2', nn.LeakyReLU(0.2, inplace=True))
        else:
            self.cnn.add_module('relu2', nn.ReLU(True))
        # Asymmetric pooling: H=(15-2)/2+1=7, W=(40-1+2*1)/1+1=41
        self.cnn.add_module('pool2', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)))

        # Layer 4: Input (B, 256, 7, 41) -> Output (B, 512, 3, 42)
        self.cnn.add_module('conv3', nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
        self.cnn.add_module('batchnorm3', nn.BatchNorm2d(512))
        if leakyRelu:
            self.cnn.add_module('relu3', nn.LeakyReLU(0.2, inplace=True))
        else:
            self.cnn.add_module('relu3', nn.ReLU(True))
        # Asymmetric pooling: H=(7-2)/2+1=3, W=(41-1+2*1)/1+1=42
        self.cnn.add_module('pool3', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)))

        # NOTE: The output of this CNN is a feature map, e.g., (Batch, 512, 3, 42) for a (Batch, nc, 60, 160) input.
        # This output needs to be reshaped and fed into an RNN/Transformer and a decoder in your downstream code.

    def forward(self, input):
        """
        Passes the input image through the CNN layers.
        Args:
            input (Tensor): Input tensor of shape (Batch, nc, H, W)
        Returns:
            Tensor: Output feature map from the CNN.
                    Shape depends on input H, W. For (B, nc, 60, 160) input,
                    output is approximately (B, 512, 3, 42).
        """
        return self.cnn(input)

def test():
    """
    Test function to check the output shape of the DdddOcr backbone.
    """
    print(f"--- Testing DdddOcr CNN Backbone ---")
    # Assuming grayscale input (nc=1) and dimensions used in CRNN design
    test_nc = 1
    test_height = IMG_HEIGHT # 60
    test_width = IMG_WIDTH   # 160
    print(f"Creating model with nc={test_nc}")
    net = DdddOcr(nc=test_nc)
    # Use a dummy batch size of 1 for testing shape
    x = torch.randn(1, test_nc, test_height, test_width)
    print(f"Input tensor shape: {x.size()}")
    y = net(x)
    print(f"Output feature map shape: {y.size()}")
    # Expected output for (1, 1, 60, 160) input based on layers: (1, 512, 3, 42)
    print(f"Expected shape for {test_height}x{test_width} input: (1, 512, 3, 42) approximately")
    print(f"------------------------------------")


if __name__ == '__main__':
    test()