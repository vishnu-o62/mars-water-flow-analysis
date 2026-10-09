from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# Load image
image = Image.open("data/raw/hirise/ESP_028535_1990.jpg")

# Convert image to tensor
transform = transforms.ToTensor()
tensor_image = transform(image)

# Print information
print("Tensor shape:", tensor_image.shape)
print("Height:", tensor_image.shape[1])
print("Width:", tensor_image.shape[2])

image = Image.open("data/raw/hirise/ESP_033216_1990.jpg")

plt.imshow(image)
plt.title("Mars Image")
plt.axis("off")
plt.show()