import cv2
import numpy as np

class ScreenshotCropper:
		def __init__(self, image_path, output_dir):
				self.image = cv2.imread(image_path)
				self.output_dir = output_dir
				self.width = 156
				self.height = 113
				self.start_x = 288
				self.start_y = 209
				self.h_spacing = 40 + self.width
				self.v_spacing = 154 + self.height								

		def crop_and_save(self):
				for row in range(3):
						for col in range(5):
								x = self.start_x + col * self.h_spacing
								y = self.start_y + row * self.v_spacing

								crop = self.image[y:y+self.height, x:x+self.width]

								cv2.imwrite(f'{self.output_dir}/card_{row}_{col}.png', crop)

				print("Cropping completed.")


# Usage example: Create an instance of the ScreenshotCropper
# cropper = ScreenshotCropper('test/Screenshot-1.png', 'tmp/screenshot')
# cropper.crop_and_save()